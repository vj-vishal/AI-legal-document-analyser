from docling.document_converter import DocumentConverter
import src.legal_rag.config as config
import re
from src.legal_rag.user_workspace.data_extraction_docling import parse_document


def extract_legal_markdown(file_path: str) -> str:
    # Initialize the converter. 
    # Under the hood, this leverages your PyTorch setup for layout models
    converter = DocumentConverter()
    
    # Process the document (handles PDFs, DOCX, etc.)
    result = converter.convert(file_path)
    
    # Export the parsed document object directly to Markdown
    markdown_content = result.document.export_to_markdown()
    
    return markdown_content

def process_legal_template(text):
    header_count = 0

    def template_logic(match):
        nonlocal header_count
        # Group 1 is the actual text after the '## '
        content = match.group(1).strip()
        
        if header_count == 0:
            # FIRST TIME: Keep '## Title' AND add 'Title' on a new line
            header_count += 1
            return f"## {content}\n{content}"
        else:
            # EVERY OTHER TIME: Just return the content (removes the ##)
            return content

    # Regex explanation:
    # ^##\s* -> Matches '##' at the start of a line and any following spaces
    # (.*)$  -> Captures the rest of the line as Group 1
    return re.sub(r'^##\s*(.*)$', template_logic, text, flags=re.MULTILINE)

if __name__ == "__main__":
    file_to_process = config.RAW_DIR/"Non-Disclosure Agreement (General).pdf"
    structured_md = extract_legal_markdown(file_to_process)

    print(structured_md)
    print("\n" + "="*80 + "\n")

    processed_md = process_legal_template(structured_md)
    print(processed_md)




