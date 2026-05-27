# UI Component definitions for PetroChat

# Dictionary to map internal source filenames to readable display names
STANDARD_NAMES = {
    "api-rp-54.pdf": "API RP 54 (Well Drilling and Servicing Safety)",
    "blm-onshore-order-2.pdf": "BLM Onshore Order No. 2 (43 CFR 3160)",
    "osha-3843.pdf": "OSHA 3843 (Safe Tank Gauging)",
    "osha-3918.pdf": "OSHA 3918 (Refinery Process Safety Management)",
    "osha-steps-alliance-citation-guide.pdf": "OSHA Steps Alliance Citation Guide"
}

def doc_list_html(docs: list, doc_chunks: dict) -> str:
    html = '<div class="kb-panel"><div style="font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; padding-left: 4px;">Loaded Documents</div>'
    for doc in docs:
        name = STANDARD_NAMES.get(doc, doc.replace(".pdf", ""))
        html += f'''<div class="kb-item" title="{name}">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2" style="flex-shrink: 0;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
    <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{name}</span>
</div>'''
    html += '</div>'
    return html

def chunk_cards_html(sources: list) -> str:
    html = '<div style="max-height: 350px; overflow-y: auto; overflow-x: hidden; padding-right: 6px;">'
    
    unique_sources = {}
    for i, s in enumerate(sources):
        src_key = s["source"]
        if src_key not in unique_sources:
            unique_sources[src_key] = []
        unique_sources[src_key].append(s)
        
    for src_key, chunks in unique_sources.items():
        name = STANDARD_NAMES.get(src_key, src_key)
        html += f'''<div style="margin-bottom: 12px; background-color: #f9f9f9; border: 1px solid #e5e5e5; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<div style="background-color: #f1f1f1; padding: 8px 12px; font-weight: 600; font-size: 0.85rem; color: #333; border-bottom: 1px solid #e5e5e5; display: flex; align-items: center; gap: 8px;">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
{name}
</div>
<div style="padding: 10px 12px; display: flex; flex-direction: column; gap: 8px;">'''
        for s in chunks:
            text = s.get("content", s.get("text", ""))
            score = s["score"]
            page = s["page"]
            
            html += f'''<div style="background-color: #ffffff; border: 1px solid #e5e5e5; border-radius: 6px; padding: 10px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 6px; align-items: center;">
<span style="font-size: 0.75rem; font-weight: 500; color: #10a37f; background-color: #e6f6f1; padding: 2px 6px; border-radius: 4px;">Page {page}</span>
<span style="font-size: 0.75rem; color: #888;">
Re-rank Score: <span style="font-weight: 600; color: {"#10a37f" if score > 0.0 else "#f59e0b"};">{score:.3f}</span>
</span>
</div>
<div style="font-size: 0.85rem; color: #444; line-height: 1.5;">{text}</div>
</div>'''
        html += '</div></div>'
    html += '</div>'
    return html
