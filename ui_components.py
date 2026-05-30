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
