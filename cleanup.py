import re
import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove render_markdown_to_html
content = re.sub(r'def render_markdown_to_html.*?return \"\\n\"\.join\(final_lines\)', '', content, flags=re.DOTALL)

# 2. Remove STANDARD_NAMES definition
content = re.sub(r'# Dictionary to map internal source filenames.*?\"osha-steps-alliance-citation-guide\.pdf\": \"OSHA Steps Alliance Citation Guide\"\n}', '', content, flags=re.DOTALL)

# 3. Remove doc_list_html and chunk_cards_html
content = re.sub(r'def doc_list_html.*?return html', '', content, flags=re.DOTALL)
content = re.sub(r'def chunk_cards_html.*?return html', '', content, flags=re.DOTALL)

# 4. Remove the huge CSS injection at the end
content = re.sub(r'# Premium Global CSS Injection.*?st\.markdown\(\"\"\"\n<style>.*?</style>\n\"\"\", unsafe_allow_html=True\)', '', content, flags=re.DOTALL)

# 5. Add the new imports
content = content.replace(
    "from groq import Groq",
    "from groq import Groq\nfrom ui_components import doc_list_html, chunk_cards_html"
)

# 6. Add style injection
page_config = """st.set_page_config(
    page_title="PetroChat",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)"""

new_page_config = """st.set_page_config(
    page_title="PetroChat",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
"""
content = content.replace(page_config, new_page_config)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
