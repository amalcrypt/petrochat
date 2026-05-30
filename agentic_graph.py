import os
import json
from typing import List, Dict, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from groq import Groq

from langgraph.graph import END, StateGraph

# Import needed functions from petrochat.py
from petrochat import perform_web_search, load_rag_resources

GROQ_MODEL = "llama-3.1-8b-instant"

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    original_query: str
    standalone_query: str
    chat_history: List[Dict[str, str]]
    
    # Execution Tracking
    plan: List[str]
    current_step_index: int
    completed_steps: List[str]
    gathered_context: List[Document]
    documents: List[Document]
    
    # State flags
    current_tool: str
    forced_fallback: bool
    
    # Generation Tracking
    generation: str
    generation_feedback: str
    iterations: int
    planner_retries: int
    missing_query: str

# --- Node Functions ---

class PetroAgent:
    def __init__(self, db, bm25):
        self.db = db
        self.bm25 = bm25
        
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
            
        self.client = Groq(api_key=api_key)

    def _structured_call(self, system_prompt: str, user_prompt: str) -> dict:
        """Helper to call Groq with JSON mode"""
        try:
            completion = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"[!] Error in structured call: {e}")
            return {}

    def route_query(self, state: GraphState) -> Literal["conversational", "plan"]:
        print("    -> [ROUTER] Analyzing query...")
        question = state.get("standalone_query") or state["original_query"]

        system = """You are an expert at routing a user question to either a 'conversational' agent or a 'plan' agent.
Use 'conversational' strictly for simple greetings like "hi", "hello", "how are you?".
For ANY technical question about oil and gas, procedures, standards, safety, or if you are unsure, use 'plan'.
You must respond with a JSON object containing a single key "datasource" with the value "conversational" or "plan"."""
        
        res = self._structured_call(system, question)
        route = res.get("datasource", "plan")
        
        if route == 'conversational':
            print("    -> [ROUTER] Routing to Conversational Node")
            return "conversational"
        else:
            print("    -> [ROUTER] Routing to Planner Node")
            return "plan"

    def reformulate(self, state: GraphState) -> GraphState:
        question = state["original_query"]
        chat_history = state.get("chat_history", [])
        
        if not chat_history:
            return {"standalone_query": question}
            
        print("    -> [REFORMULATE] Re-writing query with chat history...")
        history_str = ""
        for turn in chat_history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content']}\n"
            
        prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question (in English).
Do NOT answer the question. Only return the reformulated standalone question. If the follow-up question is already standalone or does not require history context, return it exactly as-is.

Conversation History:
{history_str}
Follow-up Question: {question}

Standalone Question:"""

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        standalone = completion.choices[0].message.content.strip()
        if standalone.startswith('"') and standalone.endswith('"'):
            standalone = standalone[1:-1]
        
        print(f"    -> [REFORMULATE] Standalone query: {standalone}")
        return {"standalone_query": standalone}
        
    def conversational_response(self, state: GraphState) -> GraphState:
        print("    -> [GENERATE] Providing conversational response...")
        question = state["original_query"]
        prompt = f"The user said: {question}\nRespond politely and professionally as PetroChat, an Oil & Gas engineering assistant. Keep it brief."
        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return {"generation": completion.choices[0].message.content}

    def plan_query(self, state: GraphState) -> GraphState:
        print("    -> [PLANNER] Decomposing query...")
        question = state.get("standalone_query") or state["original_query"]
        
        system = """You are a Planner Agent for an Oil & Gas expert system. 
Your task is to take a complex user query and decompose it into a list of simpler, independent sub-queries. 
If the user's query is simple, just return it as a single item in the list.
For comparative or complex queries, you MUST explicitly create multiple steps. For example, if asked to compare OSHA and API offshore rules:
- Step 1: Retrieve OSHA offshore rules
- Step 2: Retrieve API offshore rules
- Step 3: Compare OSHA and API rules
You must respond with a JSON object containing a single key "sub_queries" whose value is a list of strings."""

        res = self._structured_call(system, question)
        plan = res.get("sub_queries", [question])
        
        # Limit to 3 queries to avoid infinite loops
        if len(plan) > 3:
            plan = plan[:3]
            
        print(f"    -> [PLANNER] Created plan with {len(plan)} steps:")
        for idx, step in enumerate(plan):
            print(f"       Step {idx+1}: {step}")
            
        return {
            "plan": plan, 
            "current_step_index": 0,
            "completed_steps": [],
            "gathered_context": state.get("gathered_context", []),
            "iterations": 0,
            "planner_retries": state.get("planner_retries", 0)
        }

    def check_plan_complete(self, state: GraphState) -> Literal["select_tool", "generate"]:
        idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        if idx < len(plan):
            return "select_tool"
        print("    -> [PLANNER] All steps complete, moving to generation.")
        return "generate"

    def select_tool(self, state: GraphState) -> GraphState:
        idx = state["current_step_index"]
        current_query = state["plan"][idx]
        print(f"    -> [MULTI-TOOL] Selecting tool for sub-query: '{current_query}'")
        
        system = """You are a Tool Selector. For a given query, decide which tool is best:
- "vector": For semantic meaning, concepts, operations, and general Oil & Gas questions (Chroma / Uploaded PDFs).
- "bm25": For strict keyword matching, specific standard numbers (e.g. "API RP 53", "OSHA 3843") (BM25 / Uploaded PDFs).
- "web": For real-time, external, or non-technical knowledge.
You must respond with a JSON object containing a single key "tool" with value "vector", "bm25", or "web"."""
        
        res = self._structured_call(system, current_query)
        tool = res.get("tool", "vector")
        
        print(f"    -> [MULTI-TOOL] Selected '{tool}'")
        return {"current_tool": tool}

    def route_tool(self, state: GraphState) -> Literal["execute_vector", "execute_bm25", "execute_web"]:
        tool = state.get("current_tool", "vector")
        if tool == "web":
            return "execute_web"
        elif tool == "bm25":
            return "execute_bm25"
        else:
            return "execute_vector"

    def execute_vector(self, state: GraphState) -> GraphState:
        print("    -> [EXECUTE] Fetching from VectorDB...")
        idx = state["current_step_index"]
        query = state["plan"][idx]
        context = state.get("gathered_context", [])
        
        # Semantic search only
        try:
            chroma_results = self.db.similarity_search_with_score(query, k=5)
            # Deduplicate and append
            for doc, score in chroma_results:
                if not any(d.page_content == doc.page_content for d in context):
                    context.append(doc)
        except Exception as e:
            print(f"    -> [EXECUTE] VectorDB error: {e}")
                
        return {"gathered_context": context}

    def execute_bm25(self, state: GraphState) -> GraphState:
        print("    -> [EXECUTE] Fetching from BM25...")
        idx = state["current_step_index"]
        query = state["plan"][idx]
        context = state.get("gathered_context", [])
        
        if self.bm25:
            try:
                bm25_docs = self.bm25.invoke(query)
                for doc in bm25_docs:
                    if not any(d.page_content == doc.page_content for d in context):
                        context.append(doc)
            except Exception as e:
                print(f"    -> [EXECUTE] BM25 error: {e}")
                
        return {"gathered_context": context}

    def execute_web(self, state: GraphState) -> GraphState:
        idx = state["current_step_index"]
        query = state.get("missing_query") or state["plan"][idx]
        print(f"    -> [EXECUTE] Fetching from Web for query: '{query}'...")
        context = state.get("gathered_context", [])
        
        web_docs = perform_web_search(query)
        for doc in web_docs:
            if not any(d.page_content == doc.page_content for d in context):
                context.append(doc)
                
        return {"gathered_context": context}

    def reflect_on_context(self, state: GraphState) -> GraphState:
        print("    -> [REFLECTION] Reflecting on gathered context...")
        idx = state["current_step_index"]
        query = state["plan"][idx]
        context = state.get("gathered_context", [])
        
        context_str = "\n".join([d.page_content[:500] for d in context])
        
        system = """You are a Grader. Evaluate the provided context against the user query by considering:
1. Did I answer every part of the question?
2. Did I collect enough evidence?
3. What information is missing?

If information is missing, output binary_score "no" and provide a "missing_query" that specifically asks for the missing parts. If sufficient, output binary_score "yes" and "missing_query" as "".
Respond with a JSON object containing keys "binary_score" ("yes" or "no") and "missing_query" (string)."""
        
        prompt = f"Query: {query}\n\nContext: {context_str}"
        res = self._structured_call(system, prompt)
        score = res.get("binary_score", "no")
        missing_query = res.get("missing_query", query)
        
        if score == "yes" or state.get("current_tool") == "web":
            # Move to next step (If we already tried web, give up and move on to prevent loops)
            if score == "yes":
                print("    -> [REFLECTION] Context is sufficient. Moving to next step.")
            else:
                print("    -> [REFLECTION] Context insufficient, but web fallback exhausted. Moving to next step.")
            completed = state.get("completed_steps", [])
            completed.append(query)
            return {
                "completed_steps": completed,
                "current_step_index": idx + 1,
                "forced_fallback": False,
                "missing_query": ""
            }
        else:
            print(f"    -> [REFLECTION] Context insufficient. Forcing Web Search for missing info: '{missing_query}'")
            # We overwrite current_tool to web so the router pushes it there next iteration
            return {"current_tool": "web", "forced_fallback": True, "missing_query": missing_query}

    def route_after_reflection(self, state: GraphState) -> Literal["select_tool", "generate", "execute_web"]:
        if state.get("forced_fallback"):
            return "execute_web"
            
        idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        if idx < len(plan):
            return "select_tool"
            
        print("    -> [PLANNER] All steps complete, moving to generation.")
        return "generate"

    def generate(self, state: GraphState) -> GraphState:
        print("    -> [GENERATE] Crafting final answer...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("gathered_context", [])
        iterations = state.get("iterations", 0)
        feedback = state.get("generation_feedback", "")
        
        # Deduplicate and sort documents (Source Prioritization)
        # Prioritize local documents (API, OSHA, BLM) over web search
        def score_doc(doc):
            source = doc.metadata.get("source", "").lower()
            if "handbook" in source:
                return 4
            elif "web search" in source:
                return 0
            elif "api" in source:
                return 2
            elif "osha" in source:
                return 1
            return 3
            
        sorted_docs = sorted(documents, key=score_doc, reverse=True)
        
        context_str = ""
        for doc in sorted_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"
            
        # Truncate to avoid Groq 6000 TPM limits on free tier
        if len(context_str) > 12000:
            context_str = context_str[:12000] + "\n\n...[CONTEXT TRUNCATED DUE TO API LIMITS]"

        system_prompt = """You are PetroChat, an expert Oil & Gas engineer assistant specialized in drilling, production, well control, safety operations, and petroleum industry procedures.

Your responsibility is to provide 100% accurate, extremely clear, technical, and professional answers using STRICTLY the retrieved document context provided to you.

Rules for 100% Accuracy and Clarity:
1. Use ONLY information explicitly present in the retrieved context to guarantee 100% accuracy.
2. Never use your own prior knowledge. Do not guess or infer missing details.
3. Structure your answers to be exceptionally clear.
4. SOURCE PRIORITIZATION: You must heavily prioritize official standards (e.g. API RP, OSHA, BLM, Handbooks) over Web Search results. If web search contradicts an official standard, ALWAYS trust the standard.
5. Cite the source using international engineering standards referencing format (author/organization, standard identifier, and page number) after every factual statement.
   Map filenames to titles if applicable:
   - api_rp54_drilling_safety.pdf -> API RP 54 (Well Drilling and Servicing Safety)
   - osha_3843_tank_gauging.pdf -> OSHA 3843 (Safe Tank Gauging)
   - osha_3918_psm_refinery.pdf -> OSHA 3918 (Refinery Process Safety Management)
   - blm_drilling_operations.pdf -> BLM Onshore Order No. 2 (43 CFR 3160)
   - abb_production_handbook.pdf -> ABB Oil & Gas Production Handbook
   - osha_steps_citations.pdf -> OSHA Steps Alliance Citation Guide

   Example Citation: [API RP 54, Page 12]
   If Web Search, do not generate a citation block or disclaimer.

6. ALWAYS think step by step before answering. You MUST enclose your entire thought process within <think> and </think> tags. Place this block at the very beginning of your response.
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        chat_history = state.get("chat_history", [])
        for turn in chat_history[-6:]:
            messages.append(turn)

        user_content = f"Context:\n{context_str}\n\nQuestion: {question}"
        if feedback:
            user_content += f"\n\n[SYSTEM WARNING]: Your previous attempt failed verification. Reason:\n{feedback}\nPlease correct your answer."
            
        messages.append({"role": "user", "content": user_content})

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0
        )
        # Update the documents in state so that petrochat logger can log the sorted context
        return {"generation": completion.choices[0].message.content, "iterations": iterations + 1, "documents": sorted_docs}

    def grade_generation(self, state: GraphState) -> Literal["useful", "not_supported", "not_useful"]:
        print("    -> [SELF-CORRECTION] Checking generation for hallucinations...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("gathered_context", [])
        generation = state["generation"]
        iterations = state.get("iterations", 0)

        # Skip grading if we've looped too many times to prevent infinite loops
        if iterations >= 2:
            print("    -> [DECISION] Reached max iterations (2), accepting generation.")
            return "useful"
            
        retries = state.get("planner_retries", 0)
        if retries >= 3:
            print("    -> [DECISION] Reached MAX_RETRIES (3) for Planner loops. Returning best available response.")
            return "useful"

        # 1. Check Hallucinations
        context_str = ""
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"

        system_hallucination = """You are a strict evaluator assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
You must respond with a JSON object containing keys "binary_score" ("yes" or "no") and "feedback" (a brief string explaining what was hallucinated, empty if "yes")."""
        hallucination_prompt = f"Set of facts: \n\n {context_str} \n\n LLM generation: {generation}"
        
        score_h = self._structured_call(system_hallucination, hallucination_prompt)
        if score_h.get("binary_score", "no") == "no":
            feedback = score_h.get("feedback", "Answer contains hallucinations.")
            print(f"    -> [DECISION] Generation hallucinates. Feedback: {feedback}")
            self.current_feedback = feedback
            return "not_supported"

        # 2. Check Answer Relevance
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question.
You must respond with a JSON object containing a single key "binary_score" with value "yes" or "no"."""
        answer_prompt = f"User question: \n\n {question} \n\n LLM generation: {generation}"
        score_a = self._structured_call(system_answer, answer_prompt)
        
        if score_a.get("binary_score", "no") == "yes":
            print("    -> [DECISION] Generation is grounded and answers the question.")
            return "useful"
        else:
            print("    -> [DECISION] Generation is grounded but DOES NOT answer the question. Moving back to planner.")
            return "not_useful"

    def inject_feedback(self, state: GraphState) -> GraphState:
        return {"generation_feedback": getattr(self, "current_feedback", "Generation failed hallucination check.")}
        
    def record_planner_retry(self, state: GraphState) -> GraphState:
        return {"planner_retries": state.get("planner_retries", 0) + 1}


def build_graph(db, bm25):
    agent = PetroAgent(db, bm25)
    
    workflow = StateGraph(GraphState)
    
    # Nodes
    workflow.add_node("reformulate", agent.reformulate)
    workflow.add_node("plan_query", agent.plan_query)
    workflow.add_node("select_tool", agent.select_tool)
    workflow.add_node("execute_vector", agent.execute_vector)
    workflow.add_node("execute_bm25", agent.execute_bm25)
    workflow.add_node("execute_web", agent.execute_web)
    workflow.add_node("reflect_on_context", agent.reflect_on_context)
    workflow.add_node("generate", agent.generate)
    workflow.add_node("conversational", agent.conversational_response)
    workflow.add_node("inject_feedback", agent.inject_feedback)
    workflow.add_node("record_planner_retry", agent.record_planner_retry)
    
    # Edge logic
    workflow.set_entry_point("reformulate")
    
    workflow.add_conditional_edges(
        "reformulate",
        agent.route_query,
        {
            "plan": "plan_query",
            "conversational": "conversational",
        }
    )
    
    workflow.add_edge("plan_query", "select_tool")
    
    workflow.add_conditional_edges(
        "select_tool",
        agent.route_tool,
        {
            "execute_vector": "execute_vector",
            "execute_bm25": "execute_bm25",
            "execute_web": "execute_web",
        }
    )
    
    workflow.add_edge("execute_vector", "reflect_on_context")
    workflow.add_edge("execute_bm25", "reflect_on_context")
    workflow.add_edge("execute_web", "reflect_on_context")
    
    workflow.add_conditional_edges(
        "reflect_on_context",
        agent.route_after_reflection,
        {
            "execute_web": "execute_web",
            "select_tool": "select_tool",
            "generate": "generate"
        }
    )
    
    workflow.add_conditional_edges(
        "generate",
        agent.grade_generation,
        {
            "useful": END,
            "not_supported": "inject_feedback",
            "not_useful": "record_planner_retry"
        }
    )
    
    workflow.add_edge("record_planner_retry", "plan_query")
    workflow.add_edge("inject_feedback", "generate")
    workflow.add_edge("conversational", END)
    
    return workflow.compile()
