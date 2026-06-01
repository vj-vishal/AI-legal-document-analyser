import pytesseract
import pdfplumber
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
import json
import src.legal_rag.config as config

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

SUPPORTED_IMAGES       = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
SUPPORTED_PDFS         = {'.pdf'}
DIGITAL_TEXT_THRESHOLD = 50


# ── Pre-processing ───────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    img     = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh   = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


# ── Digital page check ───────────────────────────────────────────
def is_digital_page(page) -> bool:
    try:
        text = page.extract_text()
        return bool(text and len(text.strip()) > DIGITAL_TEXT_THRESHOLD)
    except Exception:
        return False


# ── Table extractor ──────────────────────────────────────────────
def extract_tables_from_page(page) -> list[str]:
    tables_markdown = []
    try:
        for table in page.extract_tables():
            if not table:
                continue
            rows = []
            for i, row in enumerate(table):
                clean = [cell.replace('\n', ' ').strip() if cell else '' for cell in row]
                rows.append('| ' + ' | '.join(clean) + ' |')
                if i == 0:
                    rows.append('| ' + ' | '.join(['---'] * len(row)) + ' |')
            tables_markdown.append('\n'.join(rows))
    except Exception:
        pass
    return tables_markdown


def get_pdf_files(directory):
    """
    Get all PDF files from a directory
    
    Args:
        directory: Path to the directory containing PDFs
    
    Returns:
        List of dictionaries with pdf_path and pdf_name
    """
    dir_path = Path(directory)
    
    # Validate directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    # Find all PDF files
    pdf_files = []
    for pdf_path in dir_path.glob("*.pdf"):
        pdf_files.append({
            "pdf_path": str(pdf_path),  # Full path as string
            "pdf_name": pdf_path.stem    # Filename with extension
        })
    
    return pdf_files



# ── Core hybrid extractor ────────────────────────────────────────
def extract_text(file_path: str) -> dict:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGES | SUPPORTED_PDFS:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {SUPPORTED_IMAGES | SUPPORTED_PDFS}")

    result = {
        "file"        : file_path.name,
        "pages"       : [],
        "method_used" : [],
        "tables"      : [],
        "errors"      : []
    }

    # ── IMAGE FILE ───────────────────────────────────────────────
    if suffix in SUPPORTED_IMAGES:
        try:
            img       = Image.open(file_path)
            processed = preprocess_image(img)
            text      = pytesseract.image_to_string(processed, config='--psm 6 --oem 1')
            result["pages"].append(text.strip())
            result["method_used"].append("tesseract")
        except Exception as e:
            result["errors"].append(f"Image OCR failed: {e}")

    # ── PDF FILE ─────────────────────────────────────────────────
    elif suffix == '.pdf':
        all_images = None

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        if is_digital_page(page):
                            text   = page.extract_text() or ""
                            tables = extract_tables_from_page(page)

                            if tables:
                                result["tables"].append({"page": page_num + 1, "data": tables})
                                text += "\n\n" + "\n\n".join(tables)

                            result["pages"].append(text.strip())
                            result["method_used"].append(f"pdfplumber (page {page_num + 1})")

                        else:
                            if all_images is None:
                                all_images = convert_from_path(str(file_path), dpi=300)

                            processed = preprocess_image(all_images[page_num])
                            text      = pytesseract.image_to_string(processed, config='--psm 6 --oem 1')
                            result["pages"].append(text.strip())
                            result["method_used"].append(f"tesseract (page {page_num + 1})")

                    except Exception as e:
                        result["errors"].append(f"Failed on page {page_num + 1}: {e}")
                        result["pages"].append("")
                        result["method_used"].append(f"failed (page {page_num + 1})")

        except Exception as e:
            result["errors"].append(f"Could not open PDF: {e}")

    return result

def enhance_structural_metadata(raw_data,category):
    
    enhanced = {
        "page_content": raw_data,
            "metadata": {
                "doc_type": "template", 
                "category": category,
                "jurisdiction": "India",
                        }
                }
        
    return enhanced


# ── Usage ────────────────────────────────────────────────────────
if __name__ == "__main__":

    pdf_list = get_pdf_files(config.RAW_DIR/"template")
    data = []

    for pdf in pdf_list:

        output = extract_text(Path(pdf["pdf_path"]))

        full_text = "\n".join(text for text in output["pages"])

        # print(f"File         : {output['file']}")
        # print(f"Methods used : {output['method_used']}")
        # print(f"Tables found : {len(output['tables'])} page(s) with tables")
        # if output["errors"]:
        #     print(f"Errors       : {output['errors']}")
        # print("-" * 60)
        # print(full_text)
        # raw_data = full_text
        enhanced_data = enhance_structural_metadata(full_text, pdf["pdf_name"])
        data.append(enhanced_data)

    with open(config.PROCESSED_DIR/"template_doc_v1.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)