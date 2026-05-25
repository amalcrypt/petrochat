import argparse
import os
import glob
import pdfplumber
import PyPDF2
import chromadb
import chromadb.config
import warnings
import pickle

# Ignore ALL warnings to keep output clean
warnings.simplefilter('ignore')

# Disable ChromaDB telemetry properly
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

def extract_text_from_pdf(pdf_path):
    """
    Extracts text page-by-page from a PDF file.
    Prefers pdfplumber, falls back to PyPDF2 for robustness.
    """
    docs = []
    filename = os.path.basename(pdf_path)
    
    # Attempt extraction with pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                # Skip pages with fewer than 50 characters (likely blank or image-only)
                if text and len(text.strip()) >= 50:
                    docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
        return docs
    except Exception as e:
        print(f"[!] pdfplumber failed for {filename}: {e}. Trying PyPDF2...")
        
    # Fallback to PyPDF2 if pdfplumber fails (e.g., due to complex encodings)
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) >= 50:
                    docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
        return docs
    except Exception as e2:
        print(f"[!] PyPDF2 also failed for {filename}: {e2}. Skipping file.")
        return []

def run_ingestion(force=False, data_dir="./data", persist_dir="./chroma_db", collection_name="petrochat_docs"):
    """
    Core ingestion function that extracts text from PDFs, splits it into chunks,
    generates vector embeddings in Chroma, and builds the BM25 index.
    Returns (success_bool, message_str).
    """
    # Gracefully handle missing data directory
    if not os.path.exists(data_dir):
        print(f"[!] Directory '{data_dir}' not found. Creating it...")
        os.makedirs(data_dir, exist_ok=True)
        print(f"    -> Please add PDFs to '{data_dir}' and run again.")
        return False, "Data directory created. Please add PDFs."

    # Check if collection exists to prevent duplicate ingestion runs
    if not force and os.path.exists(persist_dir):
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            if collection_name in [c.name for c in client.list_collections()]:
                col = client.get_collection(collection_name)
                if col.count() > 0:
                    msg = f"Collection '{collection_name}' already exists with {col.count()} documents."
                    print(msg)
                    print("Use --force to override and re-ingest.")
                    return True, msg
        except Exception as e:
            print(f"[!] Could not check existing collection: {e}")

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in '{data_dir}'.")
        return False, "No PDF files found in data/."

    all_docs = []
    processed_pdfs = 0

    for pdf_path in pdf_files:
        print(f"[+] Processing: {os.path.basename(pdf_path)}")
        docs = extract_text_from_pdf(pdf_path)
        if docs:
            all_docs.extend(docs)
            processed_pdfs += 1
            print(f"    -> Extracted {len(docs)} pages.")

    if not all_docs:
        print("[!] No text extracted from any PDFs.")
        return False, "No text could be extracted from the PDFs."

    print("[+] Splitting text into chunks...")
    # Split documents into 1000-character chunks with 200-character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    print("[+] Generating embeddings and saving to ChromaDB...")
    # Use the free HuggingFace model for embeddings (runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If forcing ingestion, wipe the old collection first
    if force and os.path.exists(persist_dir):
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            if collection_name in [c.name for c in client.list_collections()]:
                client.delete_collection(collection_name)
                print(f"[-] Deleted existing collection '{collection_name}'.")
        except Exception as e:
            print(f"[!] Warning deleting collection: {e}")

    # Create Chroma vector store and persist it to disk
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}  # Set cosine distance metric for scoring logic
    )

    print("[+] Building and saving BM25 keyword index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 15
    with open("bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

    msg = f"Ingestion complete: {processed_pdfs} PDFs | {len(chunks)} chunks stored in ChromaDB."
    print(f"[+] {msg}")
    return True, msg

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion even if DB exists.")
    args = parser.parse_args()
    run_ingestion(force=args.force)

if __name__ == "__main__":
    main()

