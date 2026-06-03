"""
PetroChat — Oil & Gas Domain RAG Assistant
Shared utilities: config constants, resource loading, logging, and web search.
"""

import os
import sys
import pickle
import warnings
import datetime
from dotenv import load_dotenv

# ── Suppress noisy output ────────────────────────────────────────────────────
warnings.simplefilter('ignore')
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except Exception:
    pass

import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Model & Path Configurations ──────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDINGS_MODEL = "BAAI/bge-large-en-v1.5"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "petrochat_docs"
BM25_PATH = "bm25_retriever.pkl"
LOG_FILE = "petrochat_session.log"

import streamlit as st

@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


# ── Resource Loading ─────────────────────────────────────────────────────────

def load_rag_resources(raise_on_missing=False):
    """
    Loads and returns (db, bm25_retriever).
    Used by app.py and evaluate.py to initialize the RAG pipeline.
    """
    if not os.path.exists(CHROMA_DIR):
        if raise_on_missing:
            raise FileNotFoundError(
                f"ChromaDB directory '{CHROMA_DIR}' not found. Please run ingestion first."
            )
        print(f"[!] ERROR: ChromaDB directory '{CHROMA_DIR}' not found. Run ingest.py first.")
        sys.exit(1)

    print("[+] Loading local embeddings model (BAAI/bge-large-en-v1.5)...")
    embeddings = load_embeddings()

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


# ── Logging ──────────────────────────────────────────────────────────────────

def log_interaction(original_query, standalone_query, retrieved_docs, final_answer):
    """
    Logs the user interaction to petrochat_session.log.
    Called by app.py after each query.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_content = []
    log_content.append(f"\n## Interaction at {timestamp}")
    log_content.append(f"* **User Query**: {original_query}")
    if standalone_query and standalone_query != original_query:
        log_content.append(f"* **Standalone Query**: {standalone_query}")
    else:
        log_content.append("* **Standalone Query**: Same as original query")

    log_content.append("\n* **Retrieved Context Chunks (Top Reranked)**:")
    for idx, (doc, score) in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        text = doc.page_content.replace("\n", " ").strip()
        log_content.append(
            f'  {idx}. [Score: {score:.4f}] {source} (Page {page}): "{text[:200]}..."'
        )

    log_content.append("\n* **Final Answer**:")
    log_content.append(f"{final_answer}")
    log_content.append("\n" + "-" * 80)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(log_content) + "\n")
    except Exception as e:
        print(f"[!] Warning: Could not write to log file: {e}")


# ── Web Search ───────────────────────────────────────────────────────────────

def perform_web_search(query):
    """
    Performs a web search using Tavily API to augment local knowledge.
    Called by agentic_graph.py when the agent decides web search is needed.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[!] TAVILY_API_KEY not found. Skipping web search.")
        return []

    try:
        from langchain_tavily import TavilySearch
        from langchain_core.documents import Document

        search = TavilySearch(max_results=2)
        res = search.invoke({"query": query})

        docs = []
        # New API returns a dict with 'results' key
        results = []
        if isinstance(res, dict):
            results = res.get("results", [])
        elif isinstance(res, list):
            results = res

        for item in results:
            content = item.get("content", "") if isinstance(item, dict) else str(item)
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": "Web Search (Tavily)", "page": "N/A"}
                ))
        return docs
    except Exception as e:
        print(f"[!] Error in web search: {e}")
    return []
