import re

def render_markdown_to_html(text):
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
                    html_lines.append("<ul style='margin-top:2px; margin-bottom:2px;'>")
                    in_sub_list = True
                html_lines.append(f"<li>{item_text}</li>")
            else:
                if in_sub_list:
                    html_lines.append("</ul>")
                    in_sub_list = False
                if not in_list:
                    html_lines.append("<ul style='margin-top:2px; margin-bottom:2px;'>")
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
        
    # Replace single newlines with <br> for plain text paragraphs (but not within lists)
    final_lines = []
    for line in html_lines:
        if line.strip() == "":
            final_lines.append("<br>")
        else:
            final_lines.append(line)
            
    return "\n".join(final_lines)

def main():
    test_text = """API RP 54 specifies several drilling safety requirements, including:
* Performing a risk assessment to determine the safe location and distance from the wellbore for pits and tanks used to circulate flammable materials [API RP 54, Page 45]
* Securing mud guns used for jetting when not in use or unattended [API RP 54, Page 45]
* Following applicable provisions for entering confined spaces when personnel need to enter a drilling fluid tank that may contain hazardous or toxic substances [API RP 54, Page 45]
* Using electric motor driven blowers with an appropriate electrical classification for the area in which they are located [API RP 54, Page 45]
* Taking precautions to prevent personnel from falling through open holes on walking surfaces of drilling fluid tanks [API RP 54, Page 45]
* Referencing other API standards and recommended practices for specific safety procedures, such as:
    + Protection against ignitions arising out of static, lightning, and stray currents [API RP 54, Page 60]
    + Safe welding, cutting, and hot work practices [API RP 54, Page 60]
    + Guidelines and procedures for entering and cleaning petroleum storage tanks [API RP 54, Page 60]
    + Safe hot tapping practices [API RP 54, Page 60]
    + Specifications for drilling and well servicing structures, wellhead and tree equipment, and drilling and production hoisting equipment [API RP 54, Page 60]
    + Standards for blowout prevention equipment systems and isolating potential flow zones during well construction [API RP 54, Page 60]"""

    html = render_markdown_to_html(test_text)
    print("--- GENERATED HTML ---")
    print(html)
    print("----------------------")

if __name__ == "__main__":
    main()
