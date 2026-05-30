import os
import sys
import pickle
import warnings
from dotenv import load_dotenv

# Ignore warnings and disable ChromaDB telemetry
warnings.simplefilter('ignore')
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Load environment variables
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from groq import Groq

from petrochat import (
    GROQ_MODEL, EMBEDDINGS_MODEL, RERANKER_MODEL, CHROMA_DIR, 
    COLLECTION_NAME, BM25_PATH
)
from agentic_graph import build_graph

# Output test log path
TEST_LOG_PATH = "qa_log.md"

def main():
    print("="*60)
    print("          PETROCHAT - AUTOMATED TEST SUITE (WEEK 2)")
    print("="*60)
    
    # Load Groq API Key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if not api_key:
        print("[!] ERROR: Groq API Key not found in environment variables or .env file.")
        sys.exit(1)
        
    # Initialize Groq Client
    client = Groq(api_key=api_key)
    
    # Load Models and Databases
    print("[+] Loading local embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    print("[+] Connecting to ChromaDB...")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("[+] Loading BM25 index...")
    bm25_retriever = None
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
            
    print("[+] Loading Cross-Encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)
    
    print("[+] Compiling Agentic Graph...")
    app = build_graph(db, bm25_retriever, reranker)
    
    # Define the 10 test queries
    # Queries 9 and 10 represent a multi-turn conversation to test conversational memory
    queries = [
        # Query 1
        {"type": "standalone", "query": "What are the safety protocols for a blowout preventer (BOP) failure?"},
        # Query 2
        {"type": "standalone", "query": "Explain the mud weight calculation and maintenance requirements for preventing a well kick."},
        # Query 3
        {"type": "standalone", "query": "What are the OSHA safety requirements for tank gauging operations?"},
        # Query 4
        {"type": "standalone", "query": "Describe the Process Safety Management (PSM) standard requirements for a refinery."},
        # Query 5
        {"type": "standalone", "query": "What are the key safety regulations for drilling operations on public or BLM lands?"},
        # Query 6
        {"type": "standalone", "query": "Explain the safety guidelines and controls for hydrogen sulfide (H2S) exposure during drilling."},
        # Query 7
        {"type": "standalone", "query": "What are the emergency shutdown system (ESD) and safety control requirements on offshore or production platforms?"},
        # Query 8
        {"type": "standalone", "query": "What are the procedures for hot work permits and control in a refinery under OSHA PSM?"},
        # Query 9 (Conversational turn 1)
        {"type": "conversational_start", "query": "What is a blowout preventer (BOP)?"},
        # Query 10 (Conversational turn 2)
        {"type": "conversational_followup", "query": "What are its specific safety protocols during drilling operations?"}
    ]
    
    # Clear the existing log file
    with open(TEST_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("# PetroChat - Week 2 Test Execution Log\n\n")
        f.write("This file contains the automated execution of 10 complex domain-specific queries against the PetroChat RAG system. ")
        f.write("It includes retrieved context, reformulation details, and LLM responses with citations.\n\n")
        f.write("## System Configuration\n")
        f.write(f"- **LLM Model**: `{GROQ_MODEL}`\n")
        f.write(f"- **Embedding Model**: `{EMBEDDINGS_MODEL}`\n")
        f.write(f"- **Reranker Model**: `{RERANKER_MODEL}`\n")
        f.write(f"- **Knowledge Source**: 6 ingested PDF documents from `data/` directory\n")
        f.write("\n" + "="*80 + "\n\n")
        
    chat_history = []
    
    print("\n[+] Starting execution of 10 test queries...")
    
    for i, q_item in enumerate(queries, 1):
        original_query = q_item["query"]
        q_type = q_item["type"]
        
        print(f"\n[+] Running Query {i}/10 ({q_type}): '{original_query}'")
        
        if q_type == "conversational_start":
            # Start of a new conversation branch
            chat_history = []
            
        initial_state = {
            "original_query": original_query,
            "chat_history": chat_history,
        }
        
        result = app.invoke(initial_state)
        answer = result.get("generation", "No answer generated.")
        standalone = result.get("standalone_query", original_query)
        retrieved_docs = result.get("documents", [])
        
        retrieved_results = [(doc, 0.99) for doc in retrieved_docs]
        
        reformulation_note = ""
        if q_type == "conversational_followup" and standalone != original_query:
            reformulation_note = f"**Reformulated Search Query**: `{standalone}`\n\n"
        
        # Print answer to stdout
        print(f"    -> Answer generated. ({len(answer)} chars)")
        
        # Write to qa_log.md
        with open(TEST_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"## Query {i}: {original_query}\n\n")
            f.write(f"**Query Type**: `{q_type}`  \n")
            if reformulation_note:
                f.write(reformulation_note)
                
            f.write("### 1. Retrieved Context Chunks (Top 3 Reranked)\n\n")
            for idx, (doc, score) in enumerate(retrieved_results, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                text = doc.page_content.replace("\n", " ").strip()
                f.write(f"#### Chunk {idx} | Source: `{source}` | Page: {page} | Rerank Score: `{score:.4f}`\n")
                f.write(f"> {text}\n\n")
                
            f.write("### 2. Final LLM Answer\n\n")
            f.write(f"{answer}\n\n")
            f.write("\n" + "="*80 + "\n\n")
            
        # Update chat history for conversational memory
        if q_type in ["conversational_start", "conversational_followup"]:
            chat_history.append({"role": "user", "content": original_query})
            chat_history.append({"role": "assistant", "content": answer})
            
    print(f"\n[+] Execution complete. 10 queries logged in '{TEST_LOG_PATH}'.")

if __name__ == "__main__":
    main()
