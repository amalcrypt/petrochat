import argparse
import os
import sys
import pickle
import warnings
import datetime
import re
from dotenv import load_dotenv

# Ignore warnings and disable ChromaDB telemetry
warnings.simplefilter('ignore')
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except Exception:
    pass

import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv()

# We need to import langchain and other packages after setting env vars
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from groq import Groq

# ── Model & Path Configurations ──────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "petrochat_docs"
BM25_PATH = "bm25_retriever.pkl"
LOG_FILE = "petrochat_session.log"


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_rag_resources(raise_on_missing=False):
    """
    Loads and returns the database, BM25 retriever, and reranker objects.
    """
    if not os.path.exists(CHROMA_DIR):
        if raise_on_missing:
            raise FileNotFoundError(f"ChromaDB directory '{CHROMA_DIR}' not found. Please run ingestion first.")
        print(f"[!] ERROR: ChromaDB directory '{CHROMA_DIR}' not found. Please run ingest.py first.")
        sys.exit(1)

    print("[+] Loading local embeddings model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    print("[+] Connecting to local Chroma database...")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("[+] Loading BM25 Keyword index...")
    bm25_retriever = None
    if os.path.exists(BM25_PATH):
        try:
            with open(BM25_PATH, "rb") as f:
                bm25_retriever = pickle.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not load BM25 index: {e}")
    else:
        print("[!] BM25 index file not found. Falling back to semantic search only.")

    print("[+] Loading Cross-Encoder Re-ranker (bge-reranker-base)...")
    reranker = CrossEncoder(RERANKER_MODEL)

    return db, bm25_retriever, reranker


def log_interaction(original_query, standalone_query, retrieved_docs, final_answer):
    """
    Logs the user interaction including original query, standalone query, 
    retrieved chunks (with scores & metadata), and the final answer.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_content = []
    log_content.append(f"\n## Interaction at {timestamp}")
    log_content.append(f"* **User Query**: {original_query}")
    if standalone_query and standalone_query != original_query:
        log_content.append(f"* **Standalone Query**: {standalone_query}")
    else:
        log_content.append(f"* **Standalone Query**: Same as original query")

    log_content.append("\n* **Retrieved Context Chunks (Top Reranked)**:")
    for idx, (doc, score) in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        text = doc.page_content.replace("\n", " ").strip()
        log_content.append(f"  {idx}. [Score: {score:.4f}] {source} (Page {page}): \"{text[:200]}...\"")

    log_content.append("\n* **Final Answer**:")
    log_content.append(f"{final_answer}")
    log_content.append("\n" + "-"*80)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(log_content) + "\n")
    except Exception as e:
        print(f"[!] Warning: Could not write to log file: {e}")


def reformulate_query(client, query, chat_history):
    """
    Uses the LLM to reformulate a follow-up query based on chat history.
    """
    if not chat_history:
        return query

    history_str = ""
    for turn in chat_history:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_str += f"{role}: {turn['content']}\n"

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question (in English) that can be used to search in a document retrieval system.
Do NOT answer the question. Only return the reformulated standalone question. If the follow-up question is already standalone or does not require history context, return it exactly as-is.

Conversation History:
{history_str}
Follow-up Question: {query}

Standalone Question:"""

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=150
        )
        standalone = completion.choices[0].message.content.strip()
        # Clean any quotes the LLM might have wrapped the response in
        if standalone.startswith('"') and standalone.endswith('"'):
            standalone = standalone[1:-1]
        return standalone
    except Exception as e:
        print(f"[!] Warning: Query reformulation failed: {e}. Using original query.")
        return query

# ── Web Search ───────────────────────────────────────────────────────────────

def perform_web_search(query):
    """
    Performs a web search using Tavily API to augment local knowledge.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[!] TAVILY_API_KEY not found in environment variables. Skipping web search.")
        return []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_core.documents import Document
        
        search = TavilySearchResults(max_results=1, tavily_api_key=api_key)
        res = search.invoke({"query": query})
        
        if res and isinstance(res, list) and len(res) > 0:
            content = res[0].get("content", "")
            if content:
                return [Document(page_content=content, metadata={"source": "Web Search (Tavily)", "page": "N/A"})]
    except Exception as e:
        print(f"[!] Error in web search: {e}")
    return []


# ── Core RAG Pipeline ────────────────────────────────────────────────────────

def retrieve_and_rerank(query, db, bm25_retriever, reranker, k_chroma=15, k_bm25=15, top_n=3, use_web_search=False):
    """
    Performs hybrid search (Chroma vector search + BM25 keyword search + optional Web Search)
    and re-ranks the combined results using a CrossEncoder.
    """
    # 1. Semantic Search
    try:
        chroma_results = db.similarity_search_with_score(query, k=k_chroma)
    except Exception as e:
        print(f"[!] Error in vector search: {e}")
        chroma_results = []

    # 2. Keyword Search
    bm25_docs = []
    if bm25_retriever:
        try:
            bm25_retriever.k = k_bm25
            bm25_docs = bm25_retriever.invoke(query)
        except Exception:
            try:
                bm25_docs = bm25_retriever.invoke(query)
            except Exception as e:
                print(f"[!] Error in BM25 search: {e}")

    # Combine and deduplicate based on page content
    unique_docs = {}
    for doc, initial_score in chroma_results:
        doc.metadata["initial_score"] = initial_score
        unique_docs[doc.page_content] = doc

    web_docs = []
    if use_web_search:
        try:
            web_docs = perform_web_search(query)
        except Exception as e:
            print(f"[!] Web search exception: {e}")

    for doc in bm25_docs + web_docs:
        if doc.page_content not in unique_docs:
            doc.metadata["initial_score"] = None
            unique_docs[doc.page_content] = doc

    combined_docs = list(unique_docs.values())

    if not combined_docs:
        return []

    # 3. Cross-Encoder Re-ranking
    pairs = [[query, doc.page_content] for doc in combined_docs]
    scores = reranker.predict(pairs)

    reranked = list(zip(combined_docs, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    top_docs = reranked[:top_n]
    
    # Ensure web search results are not filtered out by the reranker
    if web_docs:
        for w_doc in web_docs:
            if not any(d == w_doc for d, s in top_docs):
                if len(top_docs) >= top_n:
                    top_docs.pop() # Remove lowest scored local doc
                original_score = next((s for d, s in reranked if d == w_doc), 0.99)
                top_docs.insert(0, (w_doc, original_score))

    # Return top N results
    return top_docs


def verify_answer(client, query, answer, retrieved_results):
    """
    Evaluates the generated answer against the retrieved context to detect hallucinations.
    Returns (True, None) if valid, or (False, reason) if invalid.
    """
    context_str = ""
    for idx, (doc, score) in enumerate(retrieved_results, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"
        
    prompt = f"""You are a strict evaluator for an Oil & Gas engineering assistant.
Your task is to verify if the following generated answer is technically accurate and supported by the provided context (if applicable).
Ensure there are no hallucinations or unsafe engineering recommendations.

Query: {query}
Generated Answer:
{answer}

Context:
{context_str}

Evaluate the answer. If it is safe, valid, and supported, output exactly [VALID].
If it contains hallucinations or unsafe recommendations, output [INVALID] followed by the reason.
"""
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=250
        )
        eval_result = completion.choices[0].message.content.strip()
        if eval_result.startswith("[VALID]"):
            return True, None
        else:
            return False, eval_result
    except Exception as e:
        return True, None


def generate_draft_answer(client, query, retrieved_results, chat_history, feedback=None):
    """
    Queries the Groq LLM with retrieved context, enforcing strict compliance
    and hallucination prevention.
    """
    # Formulate context text
    context_str = ""
    for idx, (doc, score) in enumerate(retrieved_results, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        context_str += f"--- Document Source: {source} | Page: {page} ---\n{doc.page_content}\n\n"

    system_prompt = """
You are PetroChat, an expert Oil & Gas engineer assistant specialized in drilling, production, well control, safety operations, and petroleum industry procedures.

Your responsibility is to provide 100% accurate, extremely clear, technical, and professional answers using STRICTLY the retrieved document context provided to you.

Rules for 100% Accuracy and Clarity:

1. Use ONLY information explicitly present in the retrieved context to guarantee 100% accuracy.
   - EXCEPTION: For simple greetings (e.g., "hi", "hello"), reply conversationally and naturally without needing document context.

2. Never use your own prior knowledge, assumptions, or external information for technical questions. Do not guess or infer missing details.

3. Structure your answers to be exceptionally clear, well-organized, and easy to read.

4. When answering:
   - Be concise but technically precise.
   - Use clear Oil & Gas terminology.
   - Organize long answers using bullet points when appropriate.
   - Explain procedures step by step if available.

5. Cite the source using international engineering standards referencing format (author/organization, standard identifier, and page number) after every factual statement.
   Do NOT use raw file names in your citations. Instead, map the document filenames in the retrieved context to their respective international standard titles:
   - api_rp54_drilling_safety.pdf -> API RP 54 (Well Drilling and Servicing Safety)
   - osha_3843_tank_gauging.pdf -> OSHA 3843 (Safe Tank Gauging)
   - osha_3918_psm_refinery.pdf -> OSHA 3918 (Refinery Process Safety Management)
   - blm_drilling_operations.pdf -> BLM Onshore Order No. 2 (43 CFR 3160)
   - abb_production_handbook.pdf -> ABB Oil & Gas Production Handbook
   - osha_steps_citations.pdf -> OSHA Steps Alliance Citation Guide
   For any other PDF document not listed above, use the filename capitalized (without '.pdf' extension) as the title in the citation.

   If the information comes from a Web Search, do not generate a citation block or disclaimer. Simply state the facts naturally. Do NOT output disclaimers like "[No specific document citation is applicable here...]".

   Example Citation:
   [API RP 54, Page 12] or [OSHA 3843, Page 5]

6. Do not mention:
   - "Based on the context"
   - "The retrieved documents say"
   - "The provided information"

Present the answer naturally as an expert engineer.

7. If multiple documents provide related information:
   - Combine them logically.
   - Preserve citations for each statement.

8. Safety-related questions require maximum caution:
   - Do not invent procedures.
   - Do not create operational recommendations if not present in documents.

9. ALWAYS think step by step before answering. You MUST enclose your entire thought process within <think> and </think> tags. Place this block at the very beginning of your response. Only output your final answer after the </think> tag.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    # Limit memory to last 6 messages (3 complete turns) to prevent token bloat
    for turn in chat_history[-6:]:
        messages.append(turn)

    # Add user query with context
    user_content = f"Context:\n{context_str}\n\nQuestion: {query}"
    if feedback:
        user_content += f"\n\n[SYSTEM WARNING]: Your previous attempt failed verification. Reason:\n{feedback}\nPlease correct your answer."
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=4096
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[!] Error generating response from LLM: {e}"


def get_answer(client, query, retrieved_results, chat_history):
    """
    Generates an answer, verifies it, and regenerates if it fails verification.
    """
    # First attempt
    draft_answer = generate_draft_answer(client, query, retrieved_results, chat_history)
    
    # Verification
    is_valid, feedback = verify_answer(client, query, draft_answer, retrieved_results)
    
    if is_valid:
        return draft_answer
        
    # Second attempt with feedback
    print(f"    -> [!] Verification failed. Regenerating... Reason: {feedback}")
    second_attempt = generate_draft_answer(client, query, retrieved_results, chat_history, feedback=feedback)
    
    # Final verification
    is_valid_2, feedback_2 = verify_answer(client, query, second_attempt, retrieved_results)
    
    if is_valid_2:
        return second_attempt
    else:
        return "I don't know."


def run_rag_pipeline(query, retrieved_results):
    """
    Executes the LLM stage of the RAG pipeline using the retrieved context.
    Initializes the Groq client internally.
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if not api_key:
        return "[!] Error: Groq API Key not found in environment variables or .env file."

    try:
        client = Groq(api_key=api_key)
        return get_answer(client, query, retrieved_results, [])
    except Exception as e:
        return f"[!] Error in LLM pipeline execution: {e}"


# ── Single-Shot Mode (formerly retrieve.py) ──────────────────────────────────

def single_query_mode(query, use_web_search=True):
    """
    Runs a single query through the full RAG pipeline and prints the result.
    Equivalent to the old 'python retrieve.py "query"' command.
    """
    # Load RAG resources
    db, bm25_retriever, reranker = load_rag_resources()

    # Retrieve & Rerank — top 3 for single-shot mode
    print("Searching knowledge base & re-ranking...")
    reranked_results = retrieve_and_rerank(query, db, bm25_retriever, reranker, top_n=3, use_web_search=use_web_search)

    if not reranked_results:
        print("No relevant documents found.")
        return

    print("\n--- Top 3 Results ---")
    for i, (doc, rerank_score) in enumerate(reranked_results, 1):
        text = doc.page_content.replace("\n", " ").strip()
        if len(text) > 300:
            text = text[:300] + "..."

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        initial_score = doc.metadata.get("initial_score")

        if initial_score is not None:
            sim_score = f"{1.0 - initial_score:.2f} (Vector)"
        else:
            sim_score = "BM25"

        print(f"--- Result {i} (Rerank Score: {rerank_score:.2f} | Initial: {sim_score}) ---")
        print(f"Source: {source} | Page: {page}")
        print(f'"{text}"\n')

    # LLM Answer Generation
    print("Querying LLM pipeline...")
    answer = run_rag_pipeline(query, reranked_results)

    if answer:
        print("=" * 60)
        think_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()
            final_answer = answer.replace(think_match.group(0), "").strip()
            print("THINKING PROCESS:")
            print(think_text)
            print("-" * 60)
            print("PETROCHAT ANSWER:")
            print(final_answer)
        else:
            print("PETROCHAT ANSWER:")
            print(answer)
        print("=" * 60)


# ── Interactive Chat Mode ────────────────────────────────────────────────────

def chat_mode(use_web_search=True):
    """
    Starts the interactive multi-turn chat session.
    Equivalent to the old 'python petrochat.py' command.
    """
    print("="*60)
    print("           PETROCHAT - OIL & GAS DOMAIN RAG BOT")
    print("="*60)

    # Load Groq API Key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        print("[!] ERROR: Groq API Key not found in environment variables or .env file.")
        print("    Please set GROQ_API_KEY in your .env file.")
        sys.exit(1)

    # Initialize Groq Client
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"[!] ERROR initializing Groq client: {e}")
        sys.exit(1)

    # Load RAG resources
    db, bm25_retriever, reranker = load_rag_resources()

    print("[+] Setup complete. Conversational session active.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'clear' to reset conversation history.\n")

    # Initialize Chat History
    chat_history = []

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue

            if query.lower() == 'clear':
                chat_history = []
                print("\n[+] Conversation history cleared!\n")
                print("-" * 60)
                continue

            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            print("\n[+] Processing query...")

            # Step 1: Query Reformulation if history exists
            standalone = query
            if chat_history:
                print("    -> Reformulating query based on history...")
                standalone = reformulate_query(client, query, chat_history)
                print(f"    -> Standalone search query: '{standalone}'")

            # Step 2: Retrieve & Re-rank
            print("    -> Searching knowledge base & re-ranking...")
            retrieved_docs = retrieve_and_rerank(standalone, db, bm25_retriever, reranker, top_n=3, use_web_search=use_web_search)

            # Step 3: LLM generation
            print("    -> Querying LLM...")
            answer = get_answer(client, query, retrieved_docs, chat_history)

            # Print response
            think_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
            if think_match:
                think_text = think_match.group(1).strip()
                final_answer = answer.replace(think_match.group(0), "").strip()
                print(f"\n[Thinking Process]\n{think_text}\n")
                print(f"PetroChat:\n{final_answer}\n")
            else:
                print(f"\nPetroChat:\n{answer}\n")

            # Log the session
            log_interaction(query, standalone, retrieved_docs, answer)

            # Update Chat History
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] An error occurred during conversation: {e}\n")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PetroChat - Oil & Gas Domain RAG Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage modes:
  python petrochat.py                     Interactive chat session
  python petrochat.py --query "question"  Single-shot query & answer
"""
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query and exit (retrieve + answer mode)"
    )
    parser.add_argument(
        "--no-web",
        action="store_false",
        dest="web",
        help="Disable online web search fallback via DuckDuckGo"
    )
    args = parser.parse_args()

    if args.query:
        single_query_mode(args.query, use_web_search=args.web)
    else:
        chat_mode(use_web_search=args.web)


if __name__ == "__main__":
    main()
