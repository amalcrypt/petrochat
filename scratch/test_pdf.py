import os
import sys
from fpdf import FPDF
import datetime

class PetroChatPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(25, 195, 125) # PetroChat green (#19c37d)
        self.cell(0, 10, 'PetroChat - Oil & Gas Domain RAG Assistant', border=0, ln=1, align='L')
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

def main():
    messages = [
        {"role": "user", "content": "What is the pressure limit for separator gauge?"},
        {
            "role": "assistant",
            "content": "According to the ABB production handbook page 36, the pressure is about 4.5 MPa. Safety systems must monitor limits like HighHigh and LowLow.",
            "sources": [
                {"source": "abb_production_handbook.pdf", "page": 36, "score": 0.952, "content": "separator pressure around 4.5 MPa."}
            ]
        }
    ]
    
    pdf = PetroChatPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.set_text_color(50, 50, 50)
    
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 10, clean_txt_for_pdf("Conversation Export"), ln=1, align="L")
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8, clean_txt_for_pdf(f"Exported on: {now_str}"), ln=1, align="L")
    pdf.ln(5)
    
    for idx, msg in enumerate(messages, 1):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            pdf.set_fill_color(240, 240, 240)
            pdf.set_text_color(40, 40, 40)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 7, clean_txt_for_pdf(" USER"), ln=1, fill=True)
        else:
            pdf.set_fill_color(25, 195, 125)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 7, clean_txt_for_pdf(" PETROCHAT AI"), ln=1, fill=True)
            
        pdf.ln(2)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("helvetica", "", 10)
        
        cleaned_content = clean_txt_for_pdf(content)
        pdf.multi_cell(0, 6, cleaned_content)
        pdf.ln(5)
        
    os.makedirs("./scratch", exist_ok=True)
    output_path = "./scratch/test_export.pdf"
    pdf.output(output_path)
    print(f"PDF generated successfully at {output_path}")

if __name__ == "__main__":
    main()
