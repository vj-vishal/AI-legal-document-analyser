import pymupdf
import os

def extract_chapters_from_pdf(pdf_path):
    # Check if file exists before proceeding
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return

    # Open the document
    doc = pymupdf.open(pdf_path)

    # Extract the Table of Contents
    toc = doc.get_toc()

    if not toc:
        print("No Table of Contents (Bookmarks) found in this PDF.")
        doc.close()
        return

    print(f"Successfully loaded '{pdf_path}' ({doc.page_count} pages).\n")
    print("=" * 60)

    # Iterate through the TOC items
    for i in range(len(toc)):
        # Unpack the first three elements: hierarchy level, title, and 1-based start page
        level, title, start_page = toc[i][:3]

        # For this test, we will only extract Top-Level chapters (Level 1)
        if level == 1:
            # Determine where this chapter ends by looking at the start page of the NEXT item
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                # If it's the last item in the TOC, it ends at the last page of the document
                end_page = doc.page_count

            print(f"Chapter: {title}")
            print(f"Page Range: {start_page} to {end_page}")

            # Initialize an empty string to hold the chapter's text
            chapter_text = ""

            # Convert 1-based PDF page numbers to 0-based indices for PyMuPDF
            # We use max() and min() to ensure we don't go out of bounds
            start_idx = max(0, start_page - 1)
            end_idx = min(doc.page_count, end_page)

            # Loop through the pages of this specific chapter
            for page_num in range(start_idx, end_idx):
                page = doc.load_page(page_num)
                chapter_text += page.get_text()

            # Show a preview of the extracted text (first 150 characters) to verify the result
            clean_preview = chapter_text.strip().replace('\n', ' ')[:150]
            print(f"Extracted Preview: {clean_preview}...")
            print("-" * 60)

    doc.close()

if __name__ == "__main__":
    # Replace this string with the actual path to your PDF
    target_pdf = "data/raw/Law of Contract - I Indian Contract Act, 1872 and Specific Relief Act (S Subhashchandra, Sushilkumar Gupta) (z-library.sk, 1lib.sk, z-lib.sk).pdf"

    extract_chapters_from_pdf(target_pdf)