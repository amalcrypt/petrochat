import os
import sys

# Clean sys.path to prevent namespace package collision on Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir in sys.path:
    try:
        sys.path.remove(parent_dir)
    except ValueError:
        pass

import warnings
import logging
warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except Exception:
    pass
import traceback
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from ui_components import doc_list_html, chunk_cards_html
from petrochat import (
    load_rag_resources,
    retrieve_and_rerank,
    reformulate_query,
    get_answer,
    log_interaction,
    CHROMA_DIR,
    COLLECTION_NAME,
    BM25_PATH,
    GROQ_MODEL,
)
import json
import glob
import uuid
import time

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PetroChat",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Document Name Mapping ─────────────────────────────────────────────────────
STANDARD_NAMES = {
    "api_rp54_drilling_safety.pdf":  "API RP 54 (Well Drilling and Servicing Safety)",
    "osha_3843_tank_gauging.pdf":    "OSHA 3843 (Safe Tank Gauging)",
    "osha_3918_psm_refinery.pdf":    "OSHA 3918 (Refinery Process Safety Management)",
    "blm_drilling_operations.pdf":   "BLM Onshore Order No. 2 (43 CFR 3160)",
    "abb_production_handbook.pdf":   "ABB Oil & Gas Production Handbook",
    "osha_steps_citations.pdf":      "OSHA Steps Alliance Citation Guide",
}

# ─── PDF Generation Helpers ───────────────────────────────────────────────────
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import datetime

class PetroChatPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(25, 195, 125) # PetroChat green (#19c37d)
        self.cell(0, 10, 'PetroChat - Oil & Gas Domain RAG Assistant', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_draw_color(25, 195, 125)
        self.set_line_width(0.5)
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

def clean_txt_for_pdf(text):
    if not text:
        return ""
    replacements = {
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u2014': '-', '\u2013': '-',
        '\u2212': '-',
        '\u2022': '*',
        '\u00b0': ' degrees ',
        '\u03bc': 'micro',
        '\u2026': '...',
        '\ud83d\udcbb': '[PC]',
        '\ud83d\udee0': '[Safety]',
        '\ud83d\udcbe': '[Save]',
        '\ud83d\udccb': '[Board]',
        '\u2699': '[Config]',
        '\ud83c\udfed': '[Factory]',
        '\ud83d\udee2': '[Oil]',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf(messages):
    pdf = PetroChatPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.set_text_color(50, 50, 50)
    
    # Title
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 10, clean_txt_for_pdf("Conversation Export"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8, clean_txt_for_pdf(f"Exported on: {now_str}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
    pdf.ln(5)
    
    for idx, msg in enumerate(messages, 1):
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        
        pdf.set_line_width(0.2)
        
        # Header for the message
        if role == "user":
            pdf.set_fill_color(240, 240, 240)
            pdf.set_text_color(40, 40, 40)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 7, clean_txt_for_pdf(" USER"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        else:
            pdf.set_fill_color(25, 195, 125)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 7, clean_txt_for_pdf(" PETROCHAT AI"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            
        # Message content
        pdf.ln(2)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("helvetica", "", 10)
        
        cleaned_content = clean_txt_for_pdf(content)
        pdf.multi_cell(0, 6, cleaned_content)
        
        # Print sources if any
        if role == "assistant" and sources:
            pdf.ln(2)
            pdf.set_font("helvetica", "B", 9)
            pdf.set_text_color(25, 195, 125)
            pdf.cell(0, 5, clean_txt_for_pdf("Sources Cited:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            
            unique_docs = {}
            for s in sources:
                k = s["source"]
                if k not in unique_docs:
                    unique_docs[k] = {"pages": [], "score": s["score"]}
                unique_docs[k]["pages"].append(str(s["page"]))
            
            for raw, info in unique_docs.items():
                name = STANDARD_NAMES.get(raw, raw)
                pages = ", ".join(sorted(set(info["pages"])))
                source_line = f"  * {name} (Page {pages}) - Re-rank score: {info['score']:.3f}"
                pdf.cell(0, 5, clean_txt_for_pdf(source_line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
        pdf.ln(5)
        
    return bytes(pdf.output())



# --- 2. CUSTOM CSS OVERRIDES (ChatGPT Light Mode Exact Match) ---
# This CSS strips away Streamlit's default styling and matches the React/Tailwind design perfectly.
st.markdown('''
<style>
    /* Reset & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Default UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* App & Typography Variables */
    :root {
        --chatgpt-green: #10a37f;
        --sidebar-bg: #f9f9f9;
        --main-bg: #ffffff;
        --text-color: #374151;
        --border-color: #e5e5e5;
    }

    /* Main Backgrounds */
    .stApp {
        background-color: var(--main-bg);
    }
    
    /* Sidebar Restyling */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* New Chat Button Override */
    .stButton > button {
        width: 100%;
        background-color: #ffffff !important;
        color: #202123 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        transition: all 0.2s;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton > button:hover {
        background-color: #f9fafb !important;
        border-color: #9ca3af !important;
    }
    .stButton > button p {
        font-size: 14px;
        margin: 0;
    }

    /* Main Chat Area Padding */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 120px !important;
        max-width: 800px !important;
    }

    /* Chat Messages Restyling */
    .stChatMessage {
        padding: 1.5rem 1rem !important;
        border-radius: 0 !important;
        background-color: transparent !important;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        border-bottom: 1px solid transparent;
    }
    
    /* Target Assistant Message Background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f9f9f9 !important;
        border-top: 1px solid #f1f1f1 !important;
        border-bottom: 1px solid #f1f1f1 !important;
    }

    /* Avatar adjustments */
    .stChatMessage [data-testid="stChatAvatar"] {
        width: 2rem;
        height: 2rem;
        border-radius: 0.25rem;
    }
    
    /* Text color inside chat */
    .stChatMessage p, .stChatMessage li {
        color: #374151 !important;
        font-size: 15px !important;
        line-height: 1.75 !important;
    }

    /* Input Box Styling */
    [data-testid="stBottom"] {
        background: linear-gradient(0deg, rgba(255,255,255,1) 70%, rgba(255,255,255,0) 100%);
        padding-bottom: 20px;
    }
    .stChatInputContainer {
        border-radius: 0.75rem !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 0 15px rgba(0,0,0,0.05) !important;
        background-color: white !important;
        padding-right: 10px !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #9ca3af !important;
        box-shadow: 0 0 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Inject the disclaimer text below the input */
    [data-testid="stChatInput"]::after {
        content: "PetroChat can make mistakes. Verify critical operations data against manual logs.";
        display: block;
        text-align: center;
        font-size: 12px;
        color: #9ca3af;
        margin-top: 12px;
        font-family: 'Inter', sans-serif;
    }

    /* Custom Sidebar HTML Styles */
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
    }
    .sidebar-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.625rem 0.5rem;
        border-radius: 0.375rem;
        color: #374151;
        font-size: 0.875rem;
        cursor: pointer;
        transition: background-color 0.2s;
        text-decoration: none;
    }
    .sidebar-item:hover {
        background-color: #ececec;
    }
    
    .kb-panel {
        background-color: white;
        border: 1px solid #e5e5e5;
        border-radius: 0.5rem;
        padding: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .kb-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.375rem;
        color: #374151;
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
    }
    .kb-item:hover {
        background-color: #f9fafb;
    }
    .kb-upload-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        width: 100%;
        padding: 0.5rem;
        margin-top: 0.5rem;
        border: 1px dashed #d1d5db;
        border-radius: 0.375rem;
        color: #4b5563;
        font-size: 0.875rem;
        background: transparent;
        cursor: pointer;
    }
    .kb-upload-btn:hover {
        background-color: #f9fafb;
        border-color: #9ca3af;
    }
    
    /* Code block styling fix for light mode */
    pre code {
        color: #e5e7eb !important;
    }
</style>
''', unsafe_allow_html=True)




# ── Helpers for Assistant Cards ───────────────────────────────────────────────
def source_cards_html(sources: list) -> str:
    """Render the source citation cards below an AI answer."""
    if not sources:
        return ""
    unique_docs = {}
    for s in sources:
        k = s["source"]
        if k not in unique_docs:
            unique_docs[k] = {"pages": [], "score": s["score"]}
        unique_docs[k]["pages"].append(str(s["page"]))

    doc_items = ""
    for raw, info in unique_docs.items():
        name = STANDARD_NAMES.get(raw, raw)
        pages = ", ".join(sorted(set(info["pages"])))
        doc_items += f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 0;
            border-bottom:1px solid #3d3d3d;">
    <span style="font-size:0.9rem;">&#128196;</span>
    <div>
        <div style="font-size:0.8rem;color:#ececec;font-weight:500;">{name}</div>
        <div style="font-size:0.75rem;color:#a3a3a3;">Page {pages}</div>
    </div>
</div>"""

    return f"""
<div style="
    background:#2f2f2f;
    border:1px solid #3d3d3d;
    border-radius:12px;
    padding:14px 16px;
    margin-top:14px;
">
    <div style="font-size:0.75rem;font-weight:700;color:#19c37d;
                text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">
        &#128196; Sources
    </div>
    {doc_items}
</div>"""




def unanswerable_card(text: str) -> str:
    return f"""
<div style="
    background:rgba(239,68,68,0.1);
    border:1px solid rgba(239,68,68,0.3);
    border-left:3px solid #ef4444;
    border-radius:12px;padding:14px 16px;
    margin-top:4px;
">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-size:1rem;">&#9888;&#65039;</span>
        <span style="font-size:0.75rem;font-weight:700;color:#ef4444;
                     text-transform:uppercase;letter-spacing:0.07em;">
            Out of Knowledge Base
        </span>
    </div>
    <div style="font-size:0.875rem;color:#ececec;line-height:1.6;">{text}</div>
</div>"""




# ── Load RAG Resources ────────────────────────────────────────────────────────
@st.cache_resource
def get_rag_resources():
    try:
        db, bm25, reranker = load_rag_resources(raise_on_missing=True)
        return db, bm25, reranker, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"{e}\\n{traceback.format_exc()}"

db, bm25_retriever, reranker, init_error = get_rag_resources()

@st.cache_data
def get_database_stats(_db):
    try:
        total_chunks = _db._collection.count()
        results = _db._collection.get(include=["metadatas"])
        doc_chunks = {}
        for m in results.get("metadatas", []):
            if m and "source" in m:
                src = m["source"]
                doc_chunks[src] = doc_chunks.get(src, 0) + 1
        doc_list = sorted(doc_chunks.keys())
        return doc_list, doc_chunks, total_chunks
    except Exception:
        return [], {}, 0
env_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""
api_key = env_api_key if env_api_key else st.session_state.user_api_key


# ── Session Management Functions ──────────────────────────────────────────────
SESSIONS_DIR = "./data/sessions"

def load_saved_sessions():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    sessions = {}
    for filepath in glob.glob(os.path.join(SESSIONS_DIR, "*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "id" in data:
                    sessions[data["id"]] = data
        except Exception:
            pass
    return sessions

def save_session(session_id, title, messages):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    data = {
        "id": session_id,
        "title": title,
        "messages": messages,
        "timestamp": time.time()
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def create_new_session():
    from datetime import datetime
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.sessions[new_id] = {
        "id": new_id,
        "title": "New Chat",
        "messages": [],
        "timestamp": time.time()
    }
    st.session_state.messages = []
    save_session(new_id, "New Chat", [])
    return new_id

def delete_session(session_id):
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
    if st.session_state.current_session_id == session_id:
        if st.session_state.sessions:
            most_recent = sorted(
                st.session_state.sessions.values(),
                key=lambda s: s.get("timestamp", 0),
                reverse=True
            )[0]
            st.session_state.current_session_id = most_recent["id"]
            st.session_state.messages = most_recent["messages"]
        else:
            create_new_session()

if "sessions" not in st.session_state:
    st.session_state.sessions = load_saved_sessions()
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "clicked_prompt" not in st.session_state:
    st.session_state.clicked_prompt = None

if not st.session_state.sessions:
    create_new_session()
elif not st.session_state.current_session_id:
    most_recent = sorted(
        st.session_state.sessions.values(),
        key=lambda s: s.get("timestamp", 0),
        reverse=True
    )[0]



# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown('''
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 5px; margin-bottom: 10px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="#10a37f" stroke="#10a37f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"></path>
            </svg>
            <span style="font-size: 1.125rem; font-weight: 600; color: #202123; letter-spacing: 0.025em;">PetroChat</span>
        </div>
    ''', unsafe_allow_html=True)
    
    if st.button("➕ New Chat"):
        create_new_session()
        st.rerun()
    
    st.divider()
    st.caption("Chat History")
    # List past conversations
    sorted_sessions = sorted(st.session_state.sessions.values(), key=lambda s: s.get("timestamp", 0), reverse=True)
    for s in sorted_sessions:
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(s["title"], key=f"hist_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                st.session_state.messages = s["messages"]
                st.rerun()
        with col2:
            if st.button("✖", key=f"del_{s['id']}", help="Delete chat"):
                delete_session(s["id"])
                st.rerun()
    
    st.divider()
    st.caption("Knowledge Base")
    doc_list = []
    doc_chunks = {}
    total_chunks = 0
    
    if init_error:
        db_initialized = False
        st.markdown("""
<div style="padding:0 16px 10px;">
    <div style="
        background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
        border-radius:10px;padding:10px 12px;
        font-size:0.8rem;color:#f87171;
    ">&#9888;&#65039; Database not found. Please upload documents below to initialize it.</div>
</div>""", unsafe_allow_html=True)
    else:
        db_initialized = True
        doc_list, doc_chunks, total_chunks = get_database_stats(db)

        # Stats bar
        st.markdown(f"""
<div style="display:flex;gap:8px;padding:0 16px 12px;">
    <div style="
        flex:1;background:#2f2f2f;
        border:1px solid #3d3d3d;
        border-radius:10px;padding:8px 10px;text-align:center;
    ">
        <div style="font-size:1.1rem;font-weight:700;color:#ececec;">{len(doc_list)}</div>
        <div style="font-size:0.65rem;color:#a3a3a3;text-transform:uppercase;letter-spacing:0.05em;">Docs</div>
    </div>
    <div style="
        flex:1;background:#2f2f2f;
        border:1px solid #3d3d3d;
        border-radius:10px;padding:8px 10px;text-align:center;
    ">
        <div style="font-size:1.1rem;font-weight:700;color:#ececec;">{total_chunks}</div>
        <div style="font-size:0.65rem;color:#a3a3a3;text-transform:uppercase;letter-spacing:0.05em;">Chunks</div>
    </div>
</div>""", unsafe_allow_html=True)

        if doc_list:
            st.markdown(doc_list_html(doc_list, doc_chunks), unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding:0 16px;font-size:0.8rem;color:#a3a3a3;">No documents.</div>', unsafe_allow_html=True)

        st.markdown("<div style='padding:5px 16px 0 16px;'></div>", unsafe_allow_html=True)

    with st.expander("⚙️ Manage Documents", expanded=not db_initialized):
        # Upload Section
        st.markdown("<span style='font-size:0.8rem;font-weight:600;color:#a3a3a3;'>UPLOAD PDFs</span>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload Oil & Gas PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="doc_uploader"
        )
        if uploaded_files:
            if st.button("🚀 Ingest Documents", use_container_width=True):
                status_placeholder = st.empty()
                status_placeholder.info("Saving files to data/...")
                try:
                    os.makedirs("./data", exist_ok=True)
                    for f in uploaded_files:
                        file_path = os.path.join("./data", f.name)
                        with open(file_path, "wb") as out_f:
                            out_f.write(f.getbuffer())
                    
                    status_placeholder.info("Parsing PDFs and generating vector embeddings...")
                    from ingest import run_ingestion
                    success, msg = run_ingestion(force=True)
                    if success:
                        status_placeholder.success("🎉 Ingestion complete!")
                        st.toast("Database updated successfully!", icon="✅")
                        st.cache_resource.clear()
                        st.cache_data.clear()
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        status_placeholder.error(f"Ingestion failed: {msg}")
                except Exception as ex:
                    status_placeholder.error(f"Error: {ex}")

        st.divider()

        # Delete Section
        st.markdown("<span style='font-size:0.8rem;font-weight:600;color:#a3a3a3;'>DELETE PDF</span>", unsafe_allow_html=True)
        if doc_list:
            doc_to_delete = st.selectbox(
                "Select PDF to delete",
                options=["Select document..."] + doc_list,
                key="del_doc_select",
                label_visibility="collapsed"
            )
            if doc_to_delete != "Select document...":
                name_to_show = STANDARD_NAMES.get(doc_to_delete, doc_to_delete)
                st.warning(f"Delete '{name_to_show}'?")
                if st.button("🔴 Confirm Delete", use_container_width=True):
                    status_placeholder = st.empty()
                    status_placeholder.info("Deleting file...")
                    try:
                        file_path = os.path.join("./data", doc_to_delete)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                        # Check remaining PDF documents
                        remaining_pdfs = glob.glob(os.path.join("./data", "*.pdf"))
                        if remaining_pdfs:
                            status_placeholder.info("Re-indexing remaining documents...")
                            from ingest import run_ingestion
                            success, msg = run_ingestion(force=True)
                            if success:
                                status_placeholder.success("🎉 Deletion and re-indexing complete!")
                                st.toast("Document deleted successfully!", icon="✅")
                                st.cache_resource.clear()
                                st.cache_data.clear()
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                status_placeholder.error(f"Re-indexing failed: {msg}")
                        else:
                            # No PDFs left! Wipe database completely
                            status_placeholder.info("Wiping database (no documents left)...")
                            
                            # Wipe Chroma collection
                            try:
                                import chromadb
                                client = chromadb.PersistentClient(path=CHROMA_DIR)
                                if COLLECTION_NAME in [c.name for c in client.list_collections()]:
                                    client.delete_collection(COLLECTION_NAME)
                            except Exception as e_chroma:
                                logging.error(f"Error deleting collection: {e_chroma}")

                            # Wipe BM25 index
                            if os.path.exists(BM25_PATH):
                                try:
                                    os.remove(BM25_PATH)
                                except Exception as e_bm25:
                                    logging.error(f"Error deleting BM25 index: {e_bm25}")
                            
                            status_placeholder.success("🎉 Database wiped completely!")
                            st.toast("All documents deleted, database wiped!", icon="✅")
                            st.cache_resource.clear()
                            st.cache_data.clear()
                            time.sleep(1.5)
                            st.rerun()
                    except Exception as ex:
                        status_placeholder.error(f"Error: {ex}")
        else:
            st.markdown("<span style='font-size:0.75rem;color:#a3a3a3;'>No documents in database.</span>", unsafe_allow_html=True)

    retrieval_k = 15
    llm_n = 3

    st.divider()
    st.caption("Actions")
    if st.session_state.messages:
        try:
            pdf_data = generate_pdf(st.session_state.messages)
            st.download_button(
                label="📥 Export Chat (PDF)",
                data=pdf_data,
                file_name="petrochat_conversation.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download the current chat history as a PDF document."
            )
        except Exception as e:
            st.error(f"Error preparing PDF: {e}")

    if not env_api_key:
        st.divider()
        st.caption("Configuration")
        st.session_state.user_api_key = st.text_input(
            "🔑 Groq API Key:", 
            value=st.session_state.user_api_key, 
            type="password", 
            help="Enter your Groq API Key to start chatting."
        )
        api_key = st.session_state.user_api_key

# ── Guard: API Key & DB ────────────────────────────────────────────────────────
if not api_key:
    st.info("🔑 Please enter your Groq API Key in the sidebar to start chatting.")
    st.stop()
if not db_initialized:
    st.warning("Knowledge base not initialized. Run the ingestion script.")
    st.stop()

try:
    groq_client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

# Get prompt from chat input
prompt = st.chat_input("Ask PetroChat about Oil & Gas...")

# Handle clicked prompt or input prompt
active_prompt = None
if prompt:
    active_prompt = prompt
elif st.session_state.clicked_prompt:
    active_prompt = st.session_state.clicked_prompt
    st.session_state.clicked_prompt = None

if active_prompt:
    from datetime import datetime
    now_str = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": active_prompt, "timestamp": now_str})
    # Save user query immediately
    sid = st.session_state.current_session_id
    if sid in st.session_state.sessions:
        st.session_state.sessions[sid]["messages"] = st.session_state.messages
        if st.session_state.sessions[sid]["title"] == "New Chat":
            new_title = active_prompt[:30] + ("..." if len(active_prompt) > 30 else "")
            st.session_state.sessions[sid]["title"] = new_title
        save_session(sid, st.session_state.sessions[sid]["title"], st.session_state.messages)

# Display existing messages
if not st.session_state.messages:
    # Premium Welcome Screen
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 15vh; text-align: center; padding: 0 20px;">
        <h1 style="font-size: 3rem; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.5rem; background: -webkit-linear-gradient(45deg, #10a37f, #0c7b5f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">PetroChat</h1>
        <p style="font-size: 1.1rem; color: #6b7280; max-width: 600px; line-height: 1.5;">
            AI-powered Oil & Gas knowledge assistant for industry insights, technical documents, and intelligent conversations.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        # User doesn't use custom avatar, allowing Streamlit to assign chatAvatarIcon-user
        avatar = "💧" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            time_str = message.get("timestamp", "")
            
            if message["role"] == "user":
                st.markdown(f'<div class="msg-bubble user-bubble">{message["content"]}</div><div class="msg-time">{time_str}</div>', unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
                st.markdown(f'<div class="msg-time">{time_str}</div>', unsafe_allow_html=True)
                
            if "sources" in message and message["sources"]:
                with st.expander("▼ Retrieved Context"):
                    st.markdown(chunk_cards_html(message["sources"]), unsafe_allow_html=True)

# ─── Process RAG & Streaming for Active Prompt ────────────────────────────────
if active_prompt:
    # ── Run RAG ──
    with st.spinner(""):
        history_subset = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        standalone = active_prompt
        if history_subset:
            standalone = reformulate_query(groq_client, active_prompt, history_subset[-6:])
        
        try:
            retrieved_docs = retrieve_and_rerank(
                standalone, db, bm25_retriever, reranker,
                k_chroma=retrieval_k, k_bm25=retrieval_k, top_n=llm_n,
            )
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()
            
        if not retrieved_docs:
            no_ctx = "I cannot answer this question based on the provided documents. No relevant passages were found in the knowledge base."
            st.session_state.messages.append({"role": "assistant", "content": no_ctx, "sources": []})
            st.rerun()

        sources_info = [
            {
                "source":  doc.metadata.get("source", "Unknown"),
                "page":    doc.metadata.get("page", "?"),
                "content": doc.page_content.strip(),
                "score":   float(score),
            }
            for doc, score in retrieved_docs
        ]

        try:
            answer = get_answer(groq_client, active_prompt, retrieved_docs, history_subset[-6:])
        except Exception as e:
            st.error(f"LLM error: {e}")
            st.stop()

        log_interaction(active_prompt, standalone, retrieved_docs, answer)

    # ── Typing animation ──
    with st.chat_message("assistant", avatar="💧"):
        answer_placeholder = st.empty()
        displayed = ""
        total_len = len(answer)
        
        target_steps = 35
        chunk_size = max(1, total_len // target_steps)
        delay = 0.025 # 25ms per frame
            
        for i in range(0, total_len, chunk_size):
            displayed += answer[i:i+chunk_size]
            answer_placeholder.markdown(displayed + "▌")
            time.sleep(delay)
            
        # Final render without cursor
        answer_placeholder.markdown(answer)
        st.markdown(f'<div class="msg-time">{now_str}</div>', unsafe_allow_html=True)
        
        if sources_info:
            with st.expander("▼ Retrieved Context"):
                st.markdown(chunk_cards_html(sources_info), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_info, "timestamp": now_str})
    if sid in st.session_state.sessions:
        st.session_state.sessions[sid]["messages"] = st.session_state.messages
        save_session(sid, st.session_state.sessions[sid]["title"], st.session_state.messages)
    
    st.rerun()
