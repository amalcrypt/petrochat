import os
import warnings
import logging
warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("torch").setLevel(logging.ERROR)
import traceback
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
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
    initial_sidebar_state="collapsed",
)

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

def render_markdown_to_html(text):
    import re
    if not text:
        return ""
    
    # Convert code blocks (triple backticks)
    text = re.sub(r'```(.*?)\n(.*?)```', r'<pre style="background:#2d2d2d; padding:10px; border-radius:5px; overflow-x:auto;"><code>\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'<code style="background:#2d2d2d; padding:2px 4px; border-radius:3px;">\1</code>', text)
    
    # Convert bold
    text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<strong>\1\2</strong>', text)
    
    # Convert italic
    text = re.sub(r'\*(?!\s)(.*?)(?<!\s)\*|_(?!\s)(.*?)(?<!\s)_', r'<em>\1\2</em>', text)
    
    lines = text.split("\n")
    html_lines = []
    in_list = False
    in_sub_list = False
    
    for line in lines:
        stripped = line.strip()
        is_sub = line.startswith("    ") or line.startswith("\t") or (line.startswith("  ") and stripped.startswith(("+", "*", "-")))
        
        if stripped.startswith(("* ", "- ", "+ ")):
            item_text = stripped[2:]
            if is_sub:
                if not in_sub_list:
                    html_lines.append("<ul style='margin-top:2px; margin-bottom:2px; padding-left:20px;'>")
                    in_sub_list = True
                html_lines.append(f"<li>{item_text}</li>")
            else:
                if in_sub_list:
                    html_lines.append("</ul>")
                    in_sub_list = False
                if not in_list:
                    html_lines.append("<ul style='margin-top:2px; margin-bottom:2px; padding-left:20px;'>")
                    in_list = True
                html_lines.append(f"<li>{item_text}</li>")
        else:
            if in_sub_list:
                html_lines.append("</ul>")
                in_sub_list = False
            if in_list:
                html_lines.append("</ul>")
                in_list = True
            html_lines.append(line)
            
    if in_sub_list:
        html_lines.append("</ul>")
    if in_list:
        html_lines.append("</ul>")
        
    final_lines = []
    for line in html_lines:
        if line.strip() == "":
            final_lines.append("<br>")
        else:
            final_lines.append(line)
            
    return "\n".join(final_lines)

# ─── Global CSS — exact ChatGPT look ──────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800&display=swap');

  /* Hide Streamlit chrome */
  #MainMenu, header, footer { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Full-height dark background */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #212121 !important;
    color: #ececec;
    font-family: 'Söhne', ui-sans-serif, system-ui, -apple-system, sans-serif;
  }

  /* Sidebar — ChatGPT dark sidebar */
  [data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2d2d2d;
    padding-top: 0.5rem;
  }
  [data-testid="stSidebar"] * { color: #ececec !important; }

  /* Sidebar new-chat button */
  .new-chat-btn {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px;
    color: #ececec; font-size: 14px;
    cursor: pointer; margin-bottom: 8px;
    transition: background 0.15s;
    background: transparent;
    border: none;
    width: 100%;
    text-align: left;
  }
  .new-chat-btn:hover { background: #2d2d2d; }

  /* History buttons */
  [data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    color: #ececec !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    margin-bottom: 2px !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
  }
  [data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #2d2d2d !important;
  }

  /* Bottom chat container fix */
  [data-testid="stBottom"] {
    background-color: transparent !important;
    background: linear-gradient(transparent, #212121 30%) !important;
  }
  [data-testid="stBottom"] > div {
    background: transparent !important;
  }

  /* Main content area */
  .main .block-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    padding: 2rem 1rem 6rem !important;
  }

  /* Welcome heading */
  .welcome-heading {
    text-align: center;
    font-size: 28px;
    font-weight: 600;
    color: #ececec;
    margin: 18vh auto 2rem;
    letter-spacing: -0.3px;
  }

  /* Chat messages */
  .user-msg {
    display: flex; justify-content: flex-end;
    margin: 8px 0;
  }
  .user-bubble {
    background: #2f2f2f;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    max-width: 75%;
    font-size: 15px;
    color: #ececec;
    line-height: 1.55;
  }

  .assistant-msg {
    display: flex; align-items: flex-start; gap: 10px;
    margin: 16px 0;
  }
  .assistant-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: #19c37d;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
  }
  .assistant-bubble {
    font-size: 15px; color: #ececec;
    line-height: 1.7; padding: 4px 0;
    max-width: 90%;
  }

  /* Kill horizontal scroll globally */
  html, body, [data-testid="stAppViewContainer"], .main {
    overflow-x: hidden !important;
  }

  /* Override Streamlit chat_input */
  [data-testid="stChatInput"] {
    background: rgba(47, 47, 47, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 32px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    max-width: 760px !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
  [data-testid="stChatInput"]:focus-within {
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1) !important;
    background: rgba(47, 47, 47, 0.95) !important;
  }
  /* Only hide the inner textarea border, NOT the button */
  [data-testid="stChatInput"] div[data-baseweb="textarea"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ececec !important;
    font-size: 16px !important;
    resize: none !important;
    border: none !important;
    box-shadow: none !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: #a3a3a3 !important; }

  /* Hide native Streamlit submit button to use our custom one */
  [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {
    display: none !important;
  }

  /* Typing animation cursor */
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }
  .typing-cursor {
    display: inline-block;
    width: 2px;
    height: 15px;
    background-color: #19c37d;
    margin-left: 4px;
    animation: blink 0.8s infinite;
    vertical-align: middle;
  }

  /* Suggestion pills */
  [class*="st-key-sug_"] button {
    background: #2f2f2f !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 13.5px !important;
    color: #ececec !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    text-align: left !important;
    line-height: 1.4 !important;
    height: 100% !important;
    min-height: 85px !important;
    white-space: normal !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
  }
  [class*="st-key-sug_"] button:hover { background: #3d3d3d !important; }
  [class*="st-key-sug_"] p, [class*="st-key-sug_"] span { 
      margin: 0 !important; 
      text-align: left !important; 
      color: #ececec !important; 
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #3d3d3d; border-radius: 3px; }

</style>
""", unsafe_allow_html=True)

# Inject a custom send button into the chat input via JS
import streamlit.components.v1 as components
components.html("""
<script>
function injectSendButton() {
    var parent = window.parent.document;
    var chatInput = parent.querySelector('[data-testid="stChatInput"]');
    if (!chatInput) return;
    // Don't add if already injected
    if (chatInput.querySelector('.custom-send-btn')) return;
    
    var btn = parent.createElement('button');
    btn.className = 'custom-send-btn';
    btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ececec" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
    btn.style.cssText = 'position:absolute;right:8px;top:50%;transform:translateY(-50%);background:#3d3d3d;border:none;border-radius:8px;width:36px;height:36px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:999;transition:all 0.2s;';
    
    btn.onmouseenter = function() { btn.style.background = '#4d4d4d'; };
    btn.onmouseleave = function() { btn.style.background = '#3d3d3d'; };
    
    btn.onclick = function(e) {
        e.preventDefault();
        var textarea = chatInput.querySelector('textarea');
        if (textarea && textarea.value.trim()) {
            textarea.focus();
            var enterEvent = new KeyboardEvent('keydown', {
                key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
            });
            textarea.dispatchEvent(enterEvent);
        }
    };
    
    // Make the container relative so absolute positioning works
    chatInput.style.position = 'relative';
    chatInput.appendChild(btn);
}
setInterval(injectSendButton, 800);
setTimeout(injectSendButton, 1000);
</script>
""", height=0)


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


def chunk_cards_html(sources: list) -> str:
    html = '<div style="max-height: 350px; overflow-y: auto; overflow-x: hidden; padding-right: 6px;">'
    for i, s in enumerate(sources, 1):
        name = STANDARD_NAMES.get(s["source"], s["source"])
        score = s["score"]
        score_color = "#10b981" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#ef4444"
        preview = s["content"][:380].replace("<", "&lt;").replace(">", "&gt;")
        html += f"""
<div style="
    background:#2f2f2f;
    border:1px solid #3d3d3d;
    border-left:3px solid {score_color};
    border-radius:10px;padding:12px 14px;
    margin-bottom:10px;
    overflow-wrap: break-word;
    word-break: break-word;
">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:4px;">
        <div style="font-size:0.75rem;font-weight:600;color:#a3a3a3;">
            Chunk {i} &nbsp;&middot;&nbsp; {name} &nbsp;&middot;&nbsp; Page {s['page']}
        </div>
        <div style="font-size:0.75rem;font-weight:700;color:{score_color};
                    background:#212121;padding:2px 8px;border-radius:99px;border:1px solid #3d3d3d;">
            {score:.3f}
        </div>
    </div>
    <div style="font-size:0.8125rem;color:#ececec;line-height:1.55;overflow-wrap:break-word;word-break:break-word;">
        {preview}{"..." if len(s['content']) > 380 else ""}
    </div>
</div>"""
    html += '</div>'
    return html

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


def doc_list_html(docs: list, doc_chunks: dict) -> str:
    html = '<div style="padding:0 16px;">'
    doc_colors = ["#ececec", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
    max_chunks = max([doc_chunks.get(d, 0) for d in docs] + [1])
    for i, doc in enumerate(docs):
        name = STANDARD_NAMES.get(doc, doc.replace(".pdf", ""))
        chunks = doc_chunks.get(doc, 0)
        filepath = os.path.join("./data", doc)
        size_str = ""
        if os.path.exists(filepath):
            sz = os.path.getsize(filepath)
            size_str = f"{sz/1048576:.1f} MB" if sz > 1048576 else f"{sz/1024:.0f} KB"
        pct = int(chunks / max_chunks * 100)
        color = doc_colors[i % len(doc_colors)]
        html += f"""
<div style="
    display:flex;align-items:center;gap:10px;
    padding:8px 10px;border-radius:8px;margin-bottom:4px;
    background:#2f2f2f;border:1px solid #3d3d3d;
    transition:background 0.15s;cursor:default;
">
    <div style="
        width:6px;height:6px;border-radius:50%;
        background:{color};flex-shrink:0;
    "></div>
    <div style="flex:1;min-width:0;">
        <div style="font-size:0.8rem;color:#ececec;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:500;"
             title="{name} ({doc})">{name}</div>
        <div style="font-size:0.7rem;color:#a3a3a3;margin-top:2px;">{size_str} &nbsp;&middot;&nbsp; {chunks} chunks</div>
        <div style="width:100%;height:3px;background:#3d3d3d;border-radius:2px;margin-top:4px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};"></div>
        </div>
    </div>
</div>"""
    html += "</div>"
    return html

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
    st.session_state.current_session_id = most_recent["id"]
    st.session_state.messages = most_recent["messages"]


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True, help="Start a new chat"):
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
    if init_error:
        db_initialized = False
        st.markdown("""
<div style="padding:0 16px 10px;">
    <div style="
        background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
        border-radius:10px;padding:10px 12px;
        font-size:0.8rem;color:#f87171;
    ">&#9888;&#65039; Database not found. Run ingestion.</div>
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
        with st.expander("⚙️ Manage Documents", expanded=False):
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
prompt = st.chat_input("Ask anything about Oil & Gas...")

# Handle clicked prompt or input prompt
active_prompt = None
if prompt:
    active_prompt = prompt
elif st.session_state.clicked_prompt:
    active_prompt = st.session_state.clicked_prompt
    st.session_state.clicked_prompt = None

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    # Save user query immediately
    sid = st.session_state.current_session_id
    if sid in st.session_state.sessions:
        st.session_state.sessions[sid]["messages"] = st.session_state.messages
        if st.session_state.sessions[sid]["title"] == "New Chat":
            new_title = active_prompt[:30] + ("..." if len(active_prompt) > 30 else "")
            st.session_state.sessions[sid]["title"] = new_title
        save_session(sid, st.session_state.sessions[sid]["title"], st.session_state.messages)

# ─── Welcome screen (no messages yet) ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        '''
        <div style="text-align: center; margin-top: 12vh; margin-bottom: 3rem;">
            <h1 style="font-family: 'Montserrat', sans-serif; font-size: 4.5rem; font-weight: 800; color: #ffffff; margin-bottom: 0; letter-spacing: -1.5px; line-height: 1.1;">PetroChat AI</h1>
            <p style="font-size: 20px; font-weight: 500; color: #a3a3a3; margin-top: 1rem; letter-spacing: 0.5px;">Your smart AI companion for oil & gas knowledge, safety, and operational insights.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    prompts = [
        ("📋", "What are the safe tank gauging practices under OSHA 3843?"),
        ("⚙️", "What drilling safety requirements does API RP 54 specify?"),
        ("🏭", "Explain the Refinery Process Safety Management guidelines."),
        ("🛢️", "What does the ABB handbook say about separator operation?"),
    ]
    for idx, (icon, text) in enumerate(prompts):
        col = col1 if idx % 2 == 0 else col2
        with col:
            if st.button(f"**{icon}** {text}", key=f"sug_{idx}", use_container_width=True):
                st.session_state.clicked_prompt = text
                st.rerun()

# ─── Chat history ──────────────────────────────────────────────────────────────
else:
    st.markdown(
        '''<div style="margin-bottom: 1.5rem;">
    <span style="font-family: 'Montserrat', sans-serif; font-size: 1.6rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">PetroChat AI</span>
</div>''',
        unsafe_allow_html=True
    )
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""<div class="user-msg">
                  <div class="user-bubble">{msg["content"]}</div>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class="assistant-msg">
                  <div class="assistant-avatar">🛢️</div>
                  <div class="assistant-bubble">{render_markdown_to_html(msg["content"])}</div>
                </div>""",
                unsafe_allow_html=True
            )
            sources = msg.get("sources", [])
            if sources:
                with st.expander("▼ Retrieved Context"):
                    st.markdown(chunk_cards_html(sources), unsafe_allow_html=True)

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
    answer_placeholder = st.empty()
    displayed = ""
    total_len = len(answer)
    
    # Target around 35 steps to keep the animation duration to ~1 second, preventing WebSocket congestion
    target_steps = 35
    chunk_size = max(1, total_len // target_steps)
    delay = 0.025 # 25ms per frame
        
    for i in range(0, total_len, chunk_size):
        displayed += answer[i:i+chunk_size]
        answer_placeholder.markdown(
            f"""<div class="assistant-msg">
              <div class="assistant-avatar">🛢️</div>
              <div class="assistant-bubble">{render_markdown_to_html(displayed)}<span class="typing-cursor">|</span></div>
            </div>""",
            unsafe_allow_html=True
        )
        time.sleep(delay)
        
    # Final render without cursor
    answer_placeholder.markdown(
        f"""<div class="assistant-msg">
          <div class="assistant-avatar">🛢️</div>
          <div class="assistant-bubble">{render_markdown_to_html(answer)}</div>
        </div>""",
        unsafe_allow_html=True
    )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_info})
    if sid in st.session_state.sessions:
        st.session_state.sessions[sid]["messages"] = st.session_state.messages
        save_session(sid, st.session_state.sessions[sid]["title"], st.session_state.messages)
    
    st.rerun()
