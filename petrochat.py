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
from groq import Groq

# ── Model & Path Configurations ──────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
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

    return db, bm25_retriever


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





# ── Single-Shot Mode ─────────────────────────────────────────────────────────

def single_query_mode(query, use_web_search=True):
    """
    Runs a single query through the Agentic RAG pipeline and prints the result.
    """
    # Load RAG resources
    db, bm25_retriever = load_rag_resources()
    
    from agentic_graph import build_graph
    app = build_graph(db, bm25_retriever)

    initial_state = {
        "original_query": query,
        "chat_history": [],
    }

    print("Invoking Agentic RAG graph...")
    # Invoke the graph
    result = app.invoke(initial_state)

    answer = result.get("generation", "No answer generated.")

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
    Starts the interactive multi-turn chat session using Agentic RAG.
    """
    print("="*60)
    print("           PETROCHAT - AGENTIC OIL & GAS RAG BOT")
    print("="*60)

    # Load RAG resources
    db, bm25_retriever = load_rag_resources()

    from agentic_graph import build_graph
    app = build_graph(db, bm25_retriever)

    print("[+] Agentic Setup complete. Conversational session active.")
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

            print("\n[+] Processing query via Agentic RAG...")
            
            initial_state = {
                "original_query": query,
                "chat_history": chat_history,
            }

            result = app.invoke(initial_state)
            
            answer = result.get("generation", "Error in generation.")
            standalone = result.get("standalone_query", query)
            retrieved_docs = result.get("documents", [])
            # Fake rerank scores for the logger since graph doesn't track it currently
            docs_with_scores = [(doc, 0.99) for doc in retrieved_docs]

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
            log_interaction(query, standalone, docs_with_scores, answer)

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
