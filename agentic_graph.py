import os
import json
import re
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from groq import Groq

from langgraph.graph import END, StateGraph

# Import needed functions from petrochat.py
from petrochat import retrieve_and_rerank, perform_web_search, load_rag_resources

GROQ_MODEL = "llama-3.3-70b-versatile"

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    original_query: str
    standalone_query: str
    chat_history: List[Dict[str, str]]
    documents: List[Document]
    generation: str
    generation_feedback: str
    iterations: int

# --- Node Functions ---

class PetroAgent:
    def __init__(self, db, bm25, reranker):
        self.db = db
        self.bm25 = bm25
        self.reranker = reranker
        
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

    def route_query(self, state: GraphState) -> Literal["conversational", "vectorstore"]:
        print("    -> [ROUTER] Analyzing query...")
        question = state.get("standalone_query") or state["original_query"]

        system = """You are an expert at routing a user question to either a 'conversational' agent or a 'vectorstore'.
Use 'conversational' strictly for simple greetings like "hi", "hello", "how are you?".
For ANY technical question about oil and gas, procedures, standards, safety, or if you are unsure, use 'vectorstore'.
You must respond with a JSON object containing a single key "datasource" with the value "conversational" or "vectorstore"."""
        
        res = self._structured_call(system, question)
        route = res.get("datasource", "vectorstore")
        
        if route == 'conversational':
            print("    -> [ROUTER] Routing to Conversational Node")
            return "conversational"
        else:
            print("    -> [ROUTER] Routing to Retrieve Node")
            return "vectorstore"

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

    def retrieve(self, state: GraphState) -> GraphState:
        print("    -> [RETRIEVE] Fetching local documents...")
        question = state.get("standalone_query") or state["original_query"]
        
        # Local retrieval only first
        docs = retrieve_and_rerank(question, self.db, self.bm25, self.reranker, top_n=3, use_web_search=False)
        print(f"    -> [RETRIEVE] Retrieved {len(docs)} local chunks")
        
        # Strip out the scores for the graph state, just keep the documents
        state_docs = [doc for doc, score in docs]
        return {"documents": state_docs}

    def grade_documents(self, state: GraphState) -> GraphState:
        print("    -> [GRADER] Assessing document relevance...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("documents", [])
        
        if not documents:
            return {"documents": []}
            
        system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
You must respond with a JSON object containing a single key "binary_score" with value "yes" or "no"."""
        
        valid_docs = []
        for doc in documents:
            prompt = f"Retrieved document: \n\n {doc.page_content} \n\n User question: {question}"
            score = self._structured_call(system, prompt).get("binary_score", "no")
            if score == "yes":
                valid_docs.append(doc)
                
        print(f"    -> [GRADER] {len(valid_docs)}/{len(documents)} documents deemed relevant")
        return {"documents": valid_docs}

    def web_search(self, state: GraphState) -> GraphState:
        print("    -> [WEB SEARCH] Triggering fallback search...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("documents", [])
        
        web_docs = perform_web_search(question)
        if web_docs:
            print(f"    -> [WEB SEARCH] Retrieved {len(web_docs)} web documents")
            documents.extend(web_docs)
        else:
            print("    -> [WEB SEARCH] No web documents found")
            
        return {"documents": documents}

    def assess_relevance(self, state: GraphState) -> Literal["generate", "web_search"]:
        if not state.get("documents"):
            print("    -> [DECISION] No relevant local docs, routing to Web Search")
            return "web_search"
        print("    -> [DECISION] Local docs are relevant, routing to Generation")
        return "generate"

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

    def generate(self, state: GraphState) -> GraphState:
        print("    -> [GENERATE] Crafting final answer...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("documents", [])
        iterations = state.get("iterations", 0)
        feedback = state.get("generation_feedback", "")
        
        context_str = ""
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"

        system_prompt = """You are PetroChat, an expert Oil & Gas engineer assistant specialized in drilling, production, well control, safety operations, and petroleum industry procedures.

Your responsibility is to provide 100% accurate, extremely clear, technical, and professional answers using STRICTLY the retrieved document context provided to you.

Rules for 100% Accuracy and Clarity:
1. Use ONLY information explicitly present in the retrieved context to guarantee 100% accuracy.
2. Never use your own prior knowledge. Do not guess or infer missing details.
3. Structure your answers to be exceptionally clear.
4. Cite the source using international engineering standards referencing format (author/organization, standard identifier, and page number) after every factual statement.
   Map filenames to titles if applicable:
   - api_rp54_drilling_safety.pdf -> API RP 54 (Well Drilling and Servicing Safety)
   - osha_3843_tank_gauging.pdf -> OSHA 3843 (Safe Tank Gauging)
   - osha_3918_psm_refinery.pdf -> OSHA 3918 (Refinery Process Safety Management)
   - blm_drilling_operations.pdf -> BLM Onshore Order No. 2 (43 CFR 3160)
   - abb_production_handbook.pdf -> ABB Oil & Gas Production Handbook
   - osha_steps_citations.pdf -> OSHA Steps Alliance Citation Guide

   Example Citation: [API RP 54, Page 12]
   If Web Search, do not generate a citation block or disclaimer.

5. ALWAYS think step by step before answering. You MUST enclose your entire thought process within <think> and </think> tags. Place this block at the very beginning of your response.
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
        return {"generation": completion.choices[0].message.content, "iterations": iterations + 1}

    def grade_generation(self, state: GraphState) -> Literal["useful", "not_supported", "not_useful"]:
        print("    -> [SELF-CORRECTION] Checking generation for hallucinations...")
        question = state.get("standalone_query") or state["original_query"]
        documents = state.get("documents", [])
        generation = state["generation"]
        iterations = state.get("iterations", 0)

        # Skip grading if we've looped too many times to prevent infinite loops
        if iterations >= 2:
            print("    -> [DECISION] Reached max iterations (2), accepting generation.")
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
            print("    -> [DECISION] Generation is grounded but DOES NOT answer the question. Triggering Web Search.")
            return "not_useful"

    def inject_feedback(self, state: GraphState) -> GraphState:
        return {"generation_feedback": getattr(self, "current_feedback", "Generation failed hallucination check.")}


def build_graph(db, bm25, reranker):
    agent = PetroAgent(db, bm25, reranker)
    
    workflow = StateGraph(GraphState)
    
    # Nodes
    workflow.add_node("reformulate", agent.reformulate)
    workflow.add_node("retrieve", agent.retrieve)
    workflow.add_node("grade_documents", agent.grade_documents)
    workflow.add_node("web_search", agent.web_search)
    workflow.add_node("generate", agent.generate)
    workflow.add_node("conversational", agent.conversational_response)
    workflow.add_node("inject_feedback", agent.inject_feedback)
    
    # Edge logic
    workflow.set_entry_point("reformulate")
    
    workflow.add_conditional_edges(
        "reformulate",
        agent.route_query,
        {
            "vectorstore": "retrieve",
            "conversational": "conversational",
        }
    )
    
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        agent.assess_relevance,
        {
            "generate": "generate",
            "web_search": "web_search",
        }
    )
    
    workflow.add_edge("web_search", "generate")
    
    workflow.add_conditional_edges(
        "generate",
        agent.grade_generation,
        {
            "useful": END,
            "not_supported": "inject_feedback",
            "not_useful": "web_search"
        }
    )
    
    workflow.add_edge("inject_feedback", "generate")
    workflow.add_edge("conversational", END)
    
    return workflow.compile()
