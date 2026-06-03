"""
PetroChat — Agentic RAG Graph
A true agentic pipeline with: Query Routing, Query Decomposition (Planner),
Adaptive Tool Selection, Self-Reflection (Multi-Step Retrieval),
Context Relevance Grading, Verification (Grounding, Citations, Hallucination Checks), and Answer Checking.

Architecture:
               User Query
                     |
                     v
                   Router
                     |
       +-------------+-------------+
       |                           |
       v                           v
Conversational               Complex Query
       |                           |
       v                           v
  Generate Response             Planner
       |                           |
       v                           v
      END             +-> Multi-Tool Retrieval
                      |  (Vector/BM25/Web/API)
                      |            |
                      |            v
                      |        Reflection
                      |            |
                      |  +---------+---------+
                      |  |                   |
                      |  v                   v
            Rewrite Query<--NEED_MORE    SUFFICIENT
                      ^                      |
                      |                      v
                      |                    Grader
                      |                      |
                      |        +-------------+-------------+
                      |        |             |             |
                      +--RETRIEVE_MORE     BLOCK        PROCEED
                                             |             |
                                             v             v
                                     (Fallback Msg)    Generator
                                             |             |
                               +------------>v<------------+
                               |          Verifier --(Fail)-->+
                               |             |                |
                               +<--+      (Pass)              |
                                   |         |                |
                                   |         v                |
                                   +-<-- Answer Checker       |
                                             |                |
                                          (Pass)              |
                                             |                |
                                             v                |
                                       Final Response         |
                                                              |
                               (Feedback Loop to Generator) <-+
"""
import os
import json
import re
import time
from typing import List, Dict, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from groq import Groq
from langgraph.graph import END, StateGraph

from petrochat import perform_web_search

# ── Model Configuration ─────────────────────────────────────────────────────
# Fast model for routing, planning, grading, reflection (low-latency decisions)
GROQ_MODEL_FAST = "llama-3.1-8b-instant"
# Quality model for final answer generation
GROQ_MODEL_GEN = "llama-3.3-70b-versatile"

# ── Budget Constants ─────────────────────────────────────────────────────────
MAX_PLAN_STEPS = 3          # Max sub-queries the planner can create
MAX_RETRIES_PER_STEP = 1    # Max reflection retries before forcing SUFFICIENT
MAX_GENERATION_RETRIES = 2  # Max hallucination-check retries before accepting
MAX_RETRIEVAL_CALLS = 6     # Global budget: total retrieval calls across all steps


# ── Graph State ──────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Represents the state of the agentic RAG graph."""
    # Query
    original_query: str
    standalone_query: str
    chat_history: List[Dict[str, str]]

    # Plan execution
    plan: List[str]
    current_step_index: int
    gathered_context: List[Document]

    # Per-step tracking
    current_tool: str
    step_retries: int
    web_attempted: bool
    total_retrievals: int       # Global retrieval budget counter
    is_sufficient: bool         # Checks if context answers the original query

    # Generation & Checking
    documents: List[Document]
    generation: str
    generation_feedback: str
    iterations: int
    verification_pass: bool
    usefulness_pass: bool
    salvageable_claims: List[str]
    context_quality: Literal["SUFFICIENT", "INSUFFICIENT", "UNKNOWN"]
    known_gaps: List[str]
    verifier_issues: List[str]


# ── Agent Class ──────────────────────────────────────────────────────────────

class PetroAgent:
    def __init__(self, db, bm25):
        self.db = db
        self.bm25 = bm25

        api_key = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY")
            or os.getenv("XAI_API_KEY")
        )
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")

        self.client = Groq(api_key=api_key)
        self._last_feedback = ""

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _call_with_retry(self, **kwargs) -> any:
        """Call Groq API with automatic retries on rate limits (TPM)."""
        max_retries = 5
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate limit" in err_msg.lower():
                    # Parse wait time from message, e.g. "try again in 5.09s" or "try again in 43m58.656s"
                    wait_time = base_delay
                    
                    # 1. Look for hours, minutes, seconds format
                    time_unit_match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(\d+\.?\d*)s", err_msg)
                    if time_unit_match:
                        h = float(time_unit_match.group(1) or 0)
                        m = float(time_unit_match.group(2) or 0)
                        s = float(time_unit_match.group(3) or 0)
                        wait_time = h * 3600 + m * 60 + s + 0.5
                    else:
                        # 2. Look for simple seconds format
                        sec_match = re.search(r"try again in (\d+\.?\d*)s", err_msg)
                        if sec_match:
                            wait_time = float(sec_match.group(1)) + 0.5

                    # If the wait time is too long (e.g. Daily limit exhaustion), raise to trigger fallback
                    if wait_time > 15.0:
                        print(f"-> [SYSTEM] Rate limit wait time too long ({wait_time:.2f}s). Bypassing retry to fallback.")
                        raise e

                    print(f"-> [SYSTEM] Rate limit (429) hit. Waiting {wait_time:.2f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        return self.client.chat.completions.create(**kwargs)

    def _structured_call(self, system_prompt: str, user_prompt: str,
                         model: str = GROQ_MODEL_FAST) -> dict:
        """Call Groq with JSON mode. Returns parsed dict or {} on error, with automatic quality model fallback."""
        try:
            completion = self._call_with_retry(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"-> [SYSTEM] Error in structured call with {model}: {e}")
            if model == GROQ_MODEL_FAST:
                print(f"-> [SYSTEM] Falling back to quality model {GROQ_MODEL_GEN} for structured call...")
                try:
                    completion = self._call_with_retry(
                        model=GROQ_MODEL_GEN,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
                    return json.loads(completion.choices[0].message.content)
                except Exception as e2:
                    print(f"-> [SYSTEM] Fallback structured call also failed: {e2}")
            return {}

    # ════════════════════════════════════════════════════════════════════════
    # NODE 1: ROUTER
    # ════════════════════════════════════════════════════════════════════════

    def router_node(self, state: GraphState) -> GraphState:
        """Initial node to set up state and proceed to routing."""
        # We no longer rewrite the query, so standalone_query is just the original_query
        return {"standalone_query": state["original_query"]}

    # ════════════════════════════════════════════════════════════════════════
    # EDGE: ROUTE QUERY (Conversational vs Planner)
    # ════════════════════════════════════════════════════════════════════════

    def route_query(self, state: GraphState) -> Literal["conversational", "complex"]:
        """Route query to conversational response or multi-step planner (complex path)."""
        print("-> [ROUTER] Analyzing query...")
        question = state.get("standalone_query") or state["original_query"]

        system = """You are an expert router. Route the user question to either 'conversational' or 'complex'.
Use 'conversational' ONLY for simple greetings like "hi", "hello", "how are you?", "thanks", "goodbye".
For ANY question about oil and gas, engineering, safety, standards, procedures, or if unsure, use 'complex'.
Respond with JSON: {"route": "conversational"} or {"route": "complex"}"""

        res = self._structured_call(system, question)
        route = res.get("route", "complex")

        if route == "conversational":
            print("-> [ROUTER] Routing to Conversational")
            return "conversational"
        else:
            print("-> [ROUTER] Routing to Planner")
            return "complex"

    # ════════════════════════════════════════════════════════════════════════
    # NODE 2: CONVERSATIONAL
    # ════════════════════════════════════════════════════════════════════════

    def conversational_response(self, state: GraphState) -> GraphState:
        """Handle simple greetings and small talk."""
        print("-> [GENERATE] Providing conversational response...")
        question = state["original_query"]
        prompt = (
            f"The user said: {question}\n"
            "Respond politely and professionally as PetroChat, "
            "an Oil & Gas engineering assistant. Keep it brief."
        )
        try:
            completion = self._call_with_retry(
                model=GROQ_MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception as e:
            print(f"-> [SYSTEM] Conversational response failed with {GROQ_MODEL_FAST}: {e}. Falling back to {GROQ_MODEL_GEN}...")
            completion = self._call_with_retry(
                model=GROQ_MODEL_GEN,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        return {"generation": completion.choices[0].message.content}

    # ════════════════════════════════════════════════════════════════════════
    # NODE 3: PLAN QUERY (Query Decomposition)
    # ════════════════════════════════════════════════════════════════════════

    def plan_query(self, state: GraphState) -> GraphState:
        """Decompose the query into sub-tasks for multi-step retrieval."""
        print("-> [PLANNER] Analyzing query...")
        question = state.get("standalone_query") or state["original_query"]

        system = f"""You are a Planner Agent for an Oil & Gas expert system.
Decompose the user query into independent research sub-queries.

Rules:
- Break the question into the minimum number of sub-questions required to answer the user's request.
- Do not introduce new requirements, standards, hazards, or regulations unless explicitly required by the question.
- For simple questions, return a single sub-query (just the question itself).
- For complex questions, break into 2-{MAX_PLAN_STEPS} focused sub-queries.
- NEVER exceed {MAX_PLAN_STEPS} sub-queries.
- Each sub-query should be a broad semantic search query covering the conceptual topic, NOT a highly specific exact-match question. Vector search works better with broader queries (e.g. use "BOP failure emergency response offshore" instead of "Specific emergency procedures for BOP failure in offshore drilling").
- Use evidence-driven planning: retrieve general evidence first, then specifics.

Respond with JSON: {{"sub_queries": ["query1", "query2", ...]}}"""

        res = self._structured_call(system, question)
        plan = res.get("sub_queries", [question])

        # Hard cap at MAX_PLAN_STEPS
        if len(plan) > MAX_PLAN_STEPS:
            plan = plan[:MAX_PLAN_STEPS]
        if not plan:
            plan = [question]

        print(f"-> [PLANNER] Created plan with {len(plan)} steps:")
        for idx, step in enumerate(plan):
            print(f"   Step {idx + 1}: {step}")

        return {
            "plan": plan,
            "current_step_index": 0,
            "gathered_context": state.get("gathered_context", []),
            "iterations": 0,
            "total_retrievals": state.get("total_retrievals", 0),
            "step_retries": 0,
            "web_attempted": False,
        }

    # ════════════════════════════════════════════════════════════════════════
    # EDGE: CHECK PLAN COMPLETE
    # ════════════════════════════════════════════════════════════════════════

    def check_plan_complete(self, state: GraphState) -> Literal["select_tool", "grade_context"]:
        """Check if all plan steps are done, or if global budget is exhausted."""
        idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        total = state.get("total_retrievals", 0)

        # Global budget check
        if total >= MAX_RETRIEVAL_CALLS:
            print(f"-> [PLANNER] Global retrieval budget exhausted ({total}/{MAX_RETRIEVAL_CALLS}). Moving to grading.")
            return "grade_context"

        if idx < len(plan):
            return "select_tool"

        print("-> [PLANNER] All steps complete. Moving to grading.")
        return "grade_context"

    # ════════════════════════════════════════════════════════════════════════
    # NODE 4: SELECT TOOL (Adaptive Tool Selection)
    # ════════════════════════════════════════════════════════════════════════

    def select_tool(self, state: GraphState) -> GraphState:
        """LLM decides which retrieval tool to use for the current sub-query."""
        idx = state["current_step_index"]
        current_query = state["plan"][idx]
        print(f"-> [TOOL-SELECT] Analyzing step query: '{current_query}'")

        # Check if reflection requested a specific tool (e.g. web fallback)
        if state.get("step_retries", 0) > 0 and state.get("current_tool") == "web":
            print("-> [TOOL-SELECT] Using reflection-requested tool: web")
            return {"current_tool": "web"}

        system = """You are a Tool Selector. For a given query, decide which retrieval tool is best:
- "vector": DEFAULT. Use for semantic/conceptual questions about safety, operations, procedures, standards.
- "bm25": Use ONLY for exact keyword matches of specific document codes (e.g. "API RP 53", "OSHA 3843").
- "web": Use ONLY for real-time information, current prices, recent news, or data clearly not in internal documents.

Respond with JSON: {"tool": "vector"} or {"tool": "bm25"} or {"tool": "web"}"""

        res = self._structured_call(system, current_query)
        tool = res.get("tool", "vector")

        print(f"-> [TOOL-SELECT] Selected tool: {tool}")
        return {"current_tool": tool}

    # ════════════════════════════════════════════════════════════════════════
    # EDGE: ROUTE TO TOOL
    # ════════════════════════════════════════════════════════════════════════

    def route_tool(self, state: GraphState) -> Literal["execute_vector", "execute_bm25", "execute_web"]:
        """Route to the selected retrieval tool."""
        tool = state.get("current_tool", "vector")
        if tool == "web":
            return "execute_web"
        elif tool == "bm25":
            return "execute_bm25"
        return "execute_vector"

    # ════════════════════════════════════════════════════════════════════════
    # NODE 5a/b/c: EXECUTE RETRIEVAL TOOLS
    # ════════════════════════════════════════════════════════════════════════

    def execute_vector(self, state: GraphState) -> GraphState:
        """Semantic similarity search via ChromaDB."""
        print("-> [RETRIEVE] Running VectorDB search...")
        idx = state["current_step_index"]
        query = state["plan"][idx]
        context = list(state.get("gathered_context", []))
        total = state.get("total_retrievals", 0)

        try:
            results = self.db.similarity_search_with_score(query, k=5)
            print(f"-> [RETRIEVE] VectorDB: {len(results)} results")
            for doc, score in results:
                if not any(d.page_content == doc.page_content for d in context):
                    context.append(doc)
        except Exception as e:
            print(f"-> [RETRIEVE] VectorDB error: {e}")

        return {"gathered_context": context, "total_retrievals": total + 1}

    def execute_bm25(self, state: GraphState) -> GraphState:
        """Keyword search via BM25 index."""
        print("-> [RETRIEVE] Running BM25 search...")
        idx = state["current_step_index"]
        query = state["plan"][idx]
        context = list(state.get("gathered_context", []))
        total = state.get("total_retrievals", 0)

        if self.bm25:
            try:
                results = self.bm25.invoke(query)
                print(f"-> [RETRIEVE] BM25: {len(results)} results")
                for doc in results:
                    if not any(d.page_content == doc.page_content for d in context):
                        context.append(doc)
            except Exception as e:
                print(f"-> [RETRIEVE] BM25 error: {e}")

        return {"gathered_context": context, "total_retrievals": total + 1}

    def execute_web(self, state: GraphState) -> GraphState:
        """Web search via Tavily API."""
        print("-> [RETRIEVE] Running Web search...")
        idx = state["current_step_index"]
        query = state["plan"][idx]
        context = list(state.get("gathered_context", []))
        total = state.get("total_retrievals", 0)

        web_docs = perform_web_search(query)
        print(f"-> [RETRIEVE] Web: {len(web_docs)} results")
        for doc in web_docs:
            if not any(d.page_content == doc.page_content for d in context):
                context.append(doc)

        return {
            "gathered_context": context,
            "web_attempted": True,
            "total_retrievals": total + 1,
        }

    # ════════════════════════════════════════════════════════════════════════
    # NODE 6: REFLECT ON CONTEXT (Self-Reflection)
    # ════════════════════════════════════════════════════════════════════════

    def reflect_on_context(self, state: GraphState) -> GraphState:
        """Evaluate if retrieved context is sufficient for the current sub-query."""
        idx = state["current_step_index"]
        current_sub_query = state["plan"][idx]
        print(f"-> [REFLECT] Evaluating context for step {idx + 1}: '{current_sub_query[:60]}...'")
        context = state.get("gathered_context", [])
        step_retries = state.get("step_retries", 0)
        web_attempted = state.get("web_attempted", False)
        total = state.get("total_retrievals", 0)

        # ── HARD GUARDS (checked BEFORE LLM call to save tokens) ─────────
        # Guard 1: Global retrieval budget exhausted
        if total >= MAX_RETRIEVAL_CALLS:
            print(f"-> [REFLECT] Global budget exhausted ({total}/{MAX_RETRIEVAL_CALLS}). Forcing SUFFICIENT.")
            return {"is_sufficient": True}

        # Guard 2: Already tried web for this step — advance
        if web_attempted:
            print("-> [REFLECT] Web already attempted for this step. Advancing step.")
            return {
                "current_step_index": idx + 1,
                "step_retries": 0,
                "web_attempted": False,
                "is_sufficient": False,
            }

        # ── LLM REFLECTION (evaluate against CURRENT sub-query) ──────────
        context_str = "\n".join([d.page_content[:400] for d in context[:6]])

        system = """You are a Reflection Agent in an Agentic RAG system.
Evaluate whether the retrieved context contains relevant information for the given sub-query.

Decision Rules:
- SUFFICIENT: Context contains relevant information that addresses the sub-query (even partially).
- NEED_MORE: Context has NO relevant information for this specific sub-query.
- NEED_WEB: Information is clearly external/real-time (current prices, recent news) and not in local docs.

IMPORTANT: If the context contains ANY relevant information for the sub-query, choose SUFFICIENT.
The sub-query is just one part of a larger plan — partial coverage is acceptable.

Respond with JSON: {"decision": "SUFFICIENT" | "NEED_MORE" | "NEED_WEB", "confidence": 0.0-1.0}
where confidence represents how certain you are about your decision (1.0 = very certain, 0.0 = very uncertain)."""

        prompt = f"Sub-Query: {current_sub_query}\n\nRetrieved Context:\n{context_str}"
        res = self._structured_call(system, prompt)
        decision = res.get("decision", "SUFFICIENT")
        confidence = res.get("confidence", 0.85)

        # ── HARD GUARDS (Replacing soft guard) ──────────────────────────
        if confidence < 0.4:
            print(f"-> [REFLECT] Low confidence ({confidence}). Forcing NEED_MORE.")
            decision = "NEED_MORE"
        if len(context) < 3:
            print(f"-> [REFLECT] Too few documents ({len(context)}). Forcing NEED_MORE.")
            decision = "NEED_MORE"

        # ── Act on decision ──────────────────────────────────────────────
        if decision == "SUFFICIENT":
            # Advance to next plan step (don't early-exit to grader)
            print(f"-> [REFLECT] Step {idx + 1} context sufficient. Advancing to next step.")
            return {
                "is_sufficient": False,
                "current_step_index": idx + 1,
                "step_retries": 0,
                "web_attempted": False,
            }
        elif decision == "NEED_WEB":
            print("-> [REFLECT] Context is insufficient. Switching to Web search.")
            return {
                "is_sufficient": False,
                "current_tool": "web",
                "step_retries": step_retries + 1,
            }
        else:  # NEED_MORE
            print("-> [REFLECT] Context is insufficient. Retrieving more.")
            if step_retries >= MAX_RETRIES_PER_STEP:
                 print(f"-> [REFLECT] Step retry limit reached. Moving to next sub-query.")
                 return {
                     "is_sufficient": False,
                     "current_step_index": idx + 1,
                     "step_retries": 0,
                     "web_attempted": False,
                     "context_quality": "INSUFFICIENT"
                 }
            return {
                "is_sufficient": False,
                "step_retries": step_retries + 1,
            }

    # ════════════════════════════════════════════════════════════════════════
    # EDGE: CHECK SUFFICIENCY (Sufficient? vs Retrieve More)
    # ════════════════════════════════════════════════════════════════════════

    def check_sufficiency(self, state: GraphState) -> Literal["rewrite_query", "grade_context"]:
        """Conditional edge: check if we have enough context or need to retrieve more."""
        if state.get("is_sufficient", False):
            return "grade_context"
        
        # If not sufficient, but we've exhausted all plan steps
        if state.get("current_step_index", 0) >= len(state.get("plan", [])):
            print("-> [PLANNER] All plan steps exhausted. Forcing to Grader.")
            return "grade_context"
            
        return "rewrite_query"

    # ════════════════════════════════════════════════════════════════════════
    # NODE 6.5: REWRITE QUERY
    # ════════════════════════════════════════════════════════════════════════

    def rewrite_query(self, state: GraphState) -> GraphState:
        """Rewrite the sub-query to be broader if retrieval failed."""
        idx = state["current_step_index"]
        current_query = state["plan"][idx]
        print(f"-> [REWRITE] Broadening query: '{current_query}'")
        
        system = """You are a Query Rewriter. The previous search failed to find enough context.
Rewrite the query to be significantly broader and more conceptual. Remove specific constraints.
Respond with JSON: {"rewritten_query": "..."}"""
        
        res = self._structured_call(system, current_query)
        new_query = res.get("rewritten_query", current_query)
        
        plan = list(state["plan"])
        plan[idx] = new_query
        return {"plan": plan}

    # ════════════════════════════════════════════════════════════════════════
    # NODE 7: GRADE CONTEXT (Document Relevance Grading)
    # ════════════════════════════════════════════════════════════════════════

    def grade_context(self, state: GraphState) -> GraphState:
        """Evaluate if the gathered context contains relevant info and check coverage."""
        print("-> [GRADE] Checking document coverage...")
        question = state.get("standalone_query") or state["original_query"]
        context = state.get("gathered_context", [])
        quality = state.get("context_quality", "UNKNOWN")
        
        if quality == "INSUFFICIENT" or not context:
            print("-> [GRADE] Context quality marked INSUFFICIENT by reflection. Blocking generation.")
            return {
                "generation_feedback": "BLOCK: Retrieval quality too low.",
                "known_gaps": ["Entire query context is missing."]
            }

        # Compact context representation for grading
        context_str = "\n".join([f"[{doc.metadata.get('source', '?')}]: {doc.page_content[:300]}" for doc in context[:6]])

        system = """You are an expert document grader in an Oil & Gas domain assistant.
Evaluate the retrieved context against the user question.

Return JSON only:
{
  "coverage_score": 0.0, // 0.0 to 1.0
  "missing_aspects": ["aspect1", "aspect2"],
  "can_answer_without_fabrication": true
}"""

        prompt = f"Question: {question}\n\nRetrieved Context:\n{context_str}"
        res = self._structured_call(system, prompt)
        
        can_answer = res.get("can_answer_without_fabrication", True)
        score = res.get("coverage_score", 1.0)
        missing = res.get("missing_aspects", [])

        if not can_answer:
            print("-> [GRADE] High fabrication risk. Blocking generation.")
            return {"generation_feedback": "BLOCK: High fabrication risk.", "known_gaps": missing}
            
        if score < 0.6:
            print("-> [GRADE] Low coverage score. Generating with gaps.")
            return {"generation_feedback": "", "known_gaps": missing}

        print("-> [GRADE] Coverage sufficient. Proceeding to generation.")
        return {"generation_feedback": "", "known_gaps": missing}

    # ════════════════════════════════════════════════════════════════════════
    # NODE 8: GENERATE (Final Answer)
    # ════════════════════════════════════════════════════════════════════════

    def generate(self, state: GraphState) -> GraphState:
        """Generate the final answer using the 70B model."""
        print("-> [GENERATE] Crafting final answer...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("gathered_context", [])
        iterations = state.get("iterations", 0)
        feedback = state.get("generation_feedback", "")
        known_gaps = state.get("known_gaps", [])
        verifier_issues = state.get("verifier_issues", [])
        salvageable_claims = state.get("salvageable_claims", [])

        # Context Sufficiency Gate (from Grader Block)
        if "BLOCK:" in feedback:
            print("-> [GENERATE] Context blocked by Grader. Returning fallback.")
            return {
                "generation": "The available documents do not contain sufficient information to answer this reliably.",
                "iterations": iterations + 1,
                "documents": documents
            }

        # Sort by source priority (internal standards > web)
        def source_priority(doc):
            source = doc.metadata.get("source", "").lower()
            if "handbook" in source:
                return 6
            elif "web search" in source:
                return 1
            elif "api" in source:
                return 3
            elif "osha" in source:
                return 2
            elif "data/" in source or "blm" in source:
                return 4
            else:
                return 5  # Uploaded user documents

        sorted_docs = sorted(documents, key=source_priority, reverse=True)

        # Build context string
        context_str = ""
        for doc in sorted_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"

        # Truncate to avoid Groq TPM limits
        if len(context_str) > 12000:
            context_str = context_str[:12000] + "\n\n...[CONTEXT TRUNCATED DUE TO API LIMITS]"

        system_prompt = """You are a precise technical assistant.

VERIFIED CLAIMS (must include these):
{verified_claims}

KNOWN GAPS (acknowledge these, do not fabricate):
{known_gaps}

FORBIDDEN (instant failure):
- Do not cite page numbers
- Do not invent numeric values
- Do not reference document names not in context

Answer using ONLY the context. For each claim, you must be able to point to a specific sentence in the context above that supports it.
If a gap exists, write: "The available documents do not cover [gap]."
"""
        # Format the system prompt with state
        system_prompt = system_prompt.format(
            verified_claims="\n".join([f"- {c}" for c in salvageable_claims]) if salvageable_claims else "None",
            known_gaps="\n".join([f"- {g}" for g in known_gaps]) if known_gaps else "None"
        )

        # Build dynamic source allowlist from retrieved documents
        available_sources = set()
        for doc in sorted_docs:
            source = doc.metadata.get("source", "Unknown")
            available_sources.add(source)
        source_list = "\n".join(f"  - {s}" for s in sorted(available_sources))

        messages = [{"role": "system", "content": system_prompt}]

        chat_history = state.get("chat_history", [])
        for turn in chat_history[-6:]:
            messages.append(turn)

        user_content = (
            f"AVAILABLE SOURCE DOCUMENTS (you may ONLY reference these):\n{source_list}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}"
        )
        if verifier_issues:
            user_content += (
                f"\n\n[SYSTEM WARNING]: Your previous attempt FAILED verification. DO NOT repeat these errors:\n"
                + "\n".join([f"- {i}" for i in verifier_issues])
            )

        messages.append({"role": "user", "content": user_content})

        try:
            completion = self._call_with_retry(
                model=GROQ_MODEL_GEN,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            print(f"-> [SYSTEM] Error with generator model {GROQ_MODEL_GEN}: {e}")
            print(f"-> [SYSTEM] Falling back to fast model {GROQ_MODEL_FAST}...")
            completion = self._call_with_retry(
                model=GROQ_MODEL_FAST,
                messages=messages,
                temperature=0.0,
            )

        return {
            "generation": completion.choices[0].message.content,
            "iterations": iterations + 1,
            "documents": sorted_docs,
        }

    # ════════════════════════════════════════════════════════════════════════
    # EDGE: SELF-CORRECTION (Hallucination + Usefulness Check)
    # ════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════
    # NODE 9: VERIFIER
    # ════════════════════════════════════════════════════════════════════════

    def verify_generation(self, state: GraphState) -> GraphState:
        """Verifier node: Checks if claims are supported by context, citations are valid, and no hallucinations."""
        print("-> [VERIFIER] Verifying claims, citations, and grounding...")
        question = state.get("standalone_query") or state["original_query"]
        generation = state["generation"]
        documents = state.get("gathered_context", [])
        iterations = state.get("iterations", 0)

        past_max_retries = iterations > MAX_GENERATION_RETRIES

        # Build exact same context string as generator so verifier sees the same facts
        context_str = ""
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"
        
        if len(context_str) > 12000:
            context_str = context_str[:12000] + "\n\n...[CONTEXT TRUNCATED DUE TO API LIMITS]"

        system = """You are a Verifier for an Oil & Gas RAG system. Check if the response contains CRITICAL errors.

Focus ONLY on these critical issues:
1. FABRICATED REFERENCES: Does the response cite specific standard numbers (e.g. "API Bulletin 97"), section numbers, clause IDs, or equipment tag numbers that do NOT appear in the retrieved context? (NOTE: It is OK to cite a document without page/section numbers. Do NOT fail the response for lacking specifics).
2. INVENTED SPECIFICS: Does the response contain specific numeric values (pressures, temperatures, tolerances) not present in the context?
3. FALSE CLAIMS: Does the response make factual claims that directly contradict the retrieved context?

Do NOT flag the following:
- The absence of page numbers, section numbers, or other specific details.
- Providing partial information alongside a disclaimer.
- The disclaimer "The retrieved documents do not contain sufficient information to answer this fully."
- General professional advice (e.g. "ensure proper training", "follow safety procedures") that is a reasonable inference from the context.
- Paraphrasing or summarizing of context information.
- Minor wording differences from the source text.

Return a JSON object:
{
  "verdict": "PASS" | "FAIL",
  "issues": ["issue1", "issue2"],  // empty if PASS
  "salvageable_claims": ["claim1"] // Grounded claims. IMPORTANT: Extract the factual claim ONLY. Do NOT include citations or document names (e.g. 'is mentioned in api.pdf'). Just provide the clear, factual answer.
}"""

        prompt = f"Retrieved Context:\n{context_str}\n\nGenerated Response:\n{generation}"
        res = self._structured_call(system, prompt)

        # Fallback if API fails
        if not res:
            print("-> [VERIFIER] API error. Bypassing verification.")
            return {"generation_feedback": "", "verification_pass": True}

        verdict = res.get("verdict", "PASS")
        issues = res.get("issues", [])
        salvageable_claims = res.get("salvageable_claims", [])
        
        feedback_str = "\n".join(issues)

        if verdict == "PASS":
            print("-> [VERIFIER] Verification passed. Claims and citations are grounded.")
            return {"generation_feedback": "", "verification_pass": True, "verifier_issues": []}
        else:
            print(f"-> [VERIFIER] Verification failed: {feedback_str}")
            self._last_feedback = feedback_str
            
            # Graceful degradation on max retries
            if past_max_retries:
                print(f"-> [VERIFIER] Max retries reached ({iterations}). Degrading to partial answer.")
                if salvageable_claims:
                    final_answer = "The original answer failed verification due to potential hallucinations. Based on grounded sources, here is what can be confirmed:\n" + "\n".join(f"- {c}" for c in salvageable_claims)
                else:
                    final_answer = "Insufficient grounded information to answer the question reliably."
                # Override the generation and force pass
                return {
                    "generation": final_answer,
                    "generation_feedback": "",
                    "verification_pass": True,
                    "salvageable_claims": salvageable_claims,
                    "verifier_issues": issues
                }
            
            return {
                "generation_feedback": f"Verification failed: {feedback_str}",
                "verification_pass": False,
                "salvageable_claims": salvageable_claims,
                "verifier_issues": issues
            }

    def route_verifier(self, state: GraphState) -> Literal["check_usefulness", "inject_feedback"]:
        """Route to usefulness check if verification passes, or retry generation if it fails."""
        if state.get("verification_pass", True):
            return "check_usefulness"
        return "inject_feedback"

    # ════════════════════════════════════════════════════════════════════════
    # NODE 10: ANSWER CHECKER (USEFULNESS)
    # ════════════════════════════════════════════════════════════════════════

    def check_usefulness(self, state: GraphState) -> GraphState:
        """Answer Checker node: Checks if the generation actually answers the user's question (usefulness)."""
        print("-> [ANSWER-CHECKER] Checking query usefulness...")
        question = state.get("standalone_query") or state["original_query"]
        generation = state["generation"]
        iterations = state.get("iterations", 0)

        # Skip check after max retries
        if iterations > MAX_GENERATION_RETRIES:
            print(f"-> [ANSWER-CHECKER] Max retries reached ({iterations}). Bypassing usefulness check.")
            return {"generation_feedback": "", "usefulness_pass": True}

        # Check if it was rejected due to lack of sources - that is a useful response (guardrail rejection)
        disclaimers = [
            "i cannot answer this question",
            "not found in the available sources",
            "not found in the provided documents",
            "additional verification may be required",
            "information was not found",
            "do not contain sufficient information",
            "insufficient grounded information",
            "do not cover",
            "failed verification due to potential hallucinations"
        ]
        if any(d in generation.lower() for d in disclaimers):
            print("-> [ANSWER-CHECKER] Response is a valid out-of-context rejection. Usefulness passed.")
            return {"generation_feedback": "", "usefulness_pass": True}

        system = """You are a Usefulness Evaluator. Check if the generated response actually and fully answers the user's question.
Does it provide the relevant details asked for, or does it sidestep the prompt?

Respond in JSON format:
{"pass": true, "feedback": ""} if it answers the question.
{"pass": false, "feedback": "Detailed explanation of what part of the user's question was not answered."} if it fails."""

        prompt = f"User Question: {question}\n\nGenerated Response:\n{generation}"
        res = self._structured_call(system, prompt)

        # Fallback if API fails
        if not res:
            print("-> [ANSWER-CHECKER] API error. Bypassing usefulness check.")
            return {"generation_feedback": "", "usefulness_pass": True}

        usefulness_pass = res.get("pass", True)
        feedback = res.get("feedback", "")

        if usefulness_pass:
            print("-> [ANSWER-CHECKER] Usefulness check passed. Response is complete.")
            return {"generation_feedback": "", "usefulness_pass": True}
        else:
            print(f"-> [ANSWER-CHECKER] Usefulness check failed: {feedback}")
            self._last_feedback = feedback
            return {"generation_feedback": f"Usefulness check failed: {feedback}", "usefulness_pass": False}

    def route_usefulness(self, state: GraphState) -> Literal["accept", "fail"]:
        """Accept generation if usefulness check passes, or retry generation if it fails."""
        if state.get("usefulness_pass", True):
            return "accept"
        return "fail"

    # ── Feedback injection node ──────────────────────────────────────────

    def inject_feedback(self, state: GraphState) -> GraphState:
        """Pass failure feedback to the next generation attempt."""
        feedback = state.get("generation_feedback", self._last_feedback or "Generation failed quality checks.")
        return {"generation_feedback": feedback}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(db, bm25):
    """Build and compile the agentic RAG graph."""
    agent = PetroAgent(db, bm25)

    workflow = StateGraph(GraphState)

    # ── Register Nodes ───────────────────────────────────────────────────
    workflow.add_node("router", agent.router_node)
    workflow.add_node("conversational", agent.conversational_response)
    workflow.add_node("plan_query", agent.plan_query)
    workflow.add_node("select_tool", agent.select_tool)
    workflow.add_node("execute_vector", agent.execute_vector)
    workflow.add_node("execute_bm25", agent.execute_bm25)
    workflow.add_node("execute_web", agent.execute_web)
    workflow.add_node("reflect", agent.reflect_on_context)
    workflow.add_node("rewrite_query", agent.rewrite_query)
    workflow.add_node("grade_context", agent.grade_context)
    workflow.add_node("generate", agent.generate)
    workflow.add_node("verify_generation", agent.verify_generation)
    workflow.add_node("check_usefulness", agent.check_usefulness)
    workflow.add_node("inject_feedback", agent.inject_feedback)

    # ── Entry Point ──────────────────────────────────────────────────────
    workflow.set_entry_point("router")

    # ── Edges ────────────────────────────────────────────────────────────

    # 1. Router → Route Query (Conversational vs Complex)
    workflow.add_conditional_edges(
        "router",
        agent.route_query,
        {
            "conversational": "conversational",
            "complex": "plan_query",
        },
    )

    # 2. Conversational → END
    workflow.add_edge("conversational", END)

    # 3. Complex Path: Plan → Check if plan complete
    workflow.add_conditional_edges(
        "plan_query",
        agent.check_plan_complete,
        {
            "select_tool": "select_tool",
            "grade_context": "grade_context",
        },
    )

    # 4. Tool Selection → Execute
    workflow.add_conditional_edges(
        "select_tool",
        agent.route_tool,
        {
            "execute_vector": "execute_vector",
            "execute_bm25": "execute_bm25",
            "execute_web": "execute_web",
        },
    )

    # 5. Execution nodes → Reflect
    workflow.add_edge("execute_vector", "reflect")
    workflow.add_edge("execute_bm25", "reflect")
    workflow.add_edge("execute_web", "reflect")

    # 6. Reflect → Check Sufficiency (Sufficient? vs Retrieve More)
    workflow.add_conditional_edges(
        "reflect",
        agent.check_sufficiency,
        {
            "rewrite_query": "rewrite_query",
            "grade_context": "grade_context",
        },
    )

    # 6b. Rewrite Query → Select Tool (Retrieve More)
    workflow.add_edge("rewrite_query", "select_tool")

    # 7. Grade Context → Generate
    workflow.add_edge("grade_context", "generate")

    # 8. Generate → Verifier
    workflow.add_edge("generate", "verify_generation")

    # 9. Verifier → Route after verifier
    workflow.add_conditional_edges(
        "verify_generation",
        agent.route_verifier,
        {
            "check_usefulness": "check_usefulness",
            "inject_feedback": "inject_feedback",
        },
    )

    # 10. Check Usefulness → Route after usefulness
    workflow.add_conditional_edges(
        "check_usefulness",
        agent.route_usefulness,
        {
            "accept": END,
            "fail": "inject_feedback",
        },
    )

    # 11. Inject feedback → Regenerate
    workflow.add_edge("inject_feedback", "generate")

    return workflow.compile()
