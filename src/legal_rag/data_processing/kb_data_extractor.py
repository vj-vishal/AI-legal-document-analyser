import re
import io
import json
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.chunking import HybridChunker
from docling.datamodel.document import TextItem
from transformers import AutoTokenizer
from docling_core.types.doc import DocItemLabel

# ─────────────────────────────────────────
# Synopsis Removal Logic:
# ─────────────────────────────────────────

def remove_synopsis_block(text):
    lines = text.split('\n')
    cleaned_lines = []

    in_synopsis = False
    target_heading = None

    for line in lines:
        stripped = line.strip()

        # 1. THE TRIGGER: Enter "Delete Mode" when we see "Synopsis"
        # ^[\s|]* allows for any number of spaces or pipes before the word
        if re.match(r"(?i)^[\s|]*synopsis\b", stripped):
            in_synopsis = True
            target_heading = None # Reset target for the new synopsis
            continue

        if in_synopsis:
            # 2. THE TARGET LOCK: Figure out where to stop deleting
            if target_heading is None:
                # Skip empty lines OR Markdown table separator lines (e.g., |---|---|)
                # ^[\s|\-]+$ means "lines containing ONLY spaces, pipes, or dashes"
                if not stripped or re.match(r'^[\s|\-]+$', stripped):
                    continue

                # Chop the line into columns based on the pipes.
                # .strip('|') ensures we don't get empty strings from the outer edges of the table.
                columns = stripped.strip('|').split('|')

                # Combine the first two columns to make a strong, highly specific target
                # e.g., "[s 1.1]" + " Corresponding Section..."
                raw_target = columns[0]
                if len(columns) > 1:
                    raw_target += columns[1]

                # Normalize the combined phrase (strip punctuation, spaces, make lowercase)
                target_heading = re.sub(r'[^a-z0-9]', '', raw_target.lower())

                # Safety fallback: if target_heading is somehow completely empty, keep looking
                if not target_heading:
                    target_heading = None

                continue

            # 3. THE EXIT STRATEGY: Check if the current line is the target heading
            # Remove page indicators like [page 5], (pg 12), etc.
            # body_candidate = re.sub(r'\(?\[?(page|pg)\s*\d+\]?\)?', '', stripped.lower())
            body_candidate = re.sub(r'@@page_\d+@@', '', stripped.lower())

            # Normalize by removing all non-alphanumeric characters
            body_candidate = re.sub(r'[^a-z0-9]', '', body_candidate)

            # Check if we have a valid target, the line isn't a table of contents row (no '|'),
            # and if the body line matches our target phrase.
            if target_heading and body_candidate and "|" not in stripped:

                # We check BOTH directions just in case one is slightly longer than the other
                # (e.g., synopsis includes extra words that the body heading drops, or vice versa)
                # We enforce a minimum length of 5 characters on body_candidate to prevent false positives on tiny words.
                if body_candidate.startswith(target_heading) or (len(body_candidate) > 5 and target_heading.startswith(body_candidate)):
                    in_synopsis = False # Turn off "Delete Mode"
                    cleaned_lines.append(line) # Keep this heading line
                    continue

            # If we are in the synopsis and haven't hit the exit condition, skip this line (delete it)
            continue

        # 4. Keep normal text (when not in synopsis)
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

# ─────────────────────────────────────────
# Footnote Removal Logic:
# ─────────────────────────────────────────

def remove_ocr_citation_footnotes(text):
    # Pattern Breakdown:
    # 1. OCR Number Handling:
    #    [\[\(-]*  -> Matches optional leading brackets, parentheses, or dashes (e.g., "[", "(", "-")
    #    \s*       -> Matches optional stray spaces before the number
    #    \d+       -> Matches the actual numbers (e.g., "32")
    #    \s*       -> Matches optional spaces after the number
    #    [\]\)-]*  -> Matches optional closing brackets, parentheses, or dashes (e.g., "]", ")")
    #    \.?       -> Matches an optional dot (e.g., ".")
    #    \s+       -> Matches the space(s) before the actual text begins
    #
    # 2. Safety Bounds & Keywords:
    #    (?:(?!@@PAGE).)*?                          -> Safely scans text without jumping past a page tag
    #    \b(?:v|vs|AIR|SCC|Cr LJ|SC|All LJ|Edn|Ker)\b   -> MUST contain a citation keyword ("v" or "vs" are great anchors)
    #    (?:(?!@@PAGE).)*?                          -> Matches the rest of the citation
    #    @@PAGE_\d+@@\s*                            -> Ends strictly at the page marker
    
    pattern = r"[\[\(-]*\s*\d+\s*[\]\)-]*\.?\s+(?:(?!@@PAGE).)*?\b(?:v|vs|AIR|SCC|Cr LJ|SC|All LJ|Edn|Ker)\b(?:(?!@@PAGE).)*?@@PAGE_\d+@@"
    
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    return cleaned_text.strip()

# ─────────────────────────────────────────
# Regex Engine Compilation
# ─────────────────────────────────────────

def create_universal_regex(raw_toc_string):
    """
    Converts a TOC string into a hyper-flexible regex object that ignores
    OCR prefix garbage, fractured spaces, and corrupted quotes.
    """
    if not raw_toc_string:
        return None

    # 1. Escape regex control operators
    escaped_text = re.escape(str(raw_toc_string).strip())

    # 2. Flexible Spaces: Replace "\ " with 1 or more spaces
    flex = re.sub(r'\\ ', r'\\s+', escaped_text)

    # 3. Flexible Dots: Allows "5." or "5 ." or "5  ."
    flex = re.sub(r'\\\.', r'\\s*\\.\\s*', flex)

    # 4. Universal Quotes/Apostrophes: Allows "Purchaser's" to match "Purchaser ' s" or "Purchaser ` s"
    # Matches any single or double quote, curly or straight, with optional spaces around it
    flex = re.sub(r'\\\'|\\\"|\\”|\\“|\\’|\\‘', r'\\s*[\'\"”“’‘`]\\s*', flex)

    # 5. Flexible Punctuation: Allows "uncertain." to match "uncertain ."
    flex = re.sub(r'\\\,', r'\\s*\,\\s*', flex)
    flex = re.sub(r'\\\:', r'\\s*\\:\\s*', flex)
    flex = re.sub(r'\\\-', r'\\s*[\\-\\—\\–]\\s*', flex)

    # --- THE BOUNDARIES ---
    # PREFIX GOBBLER: [^A-Za-z]{1,10}
    # Eats up to 10 non-alphabet characters at the start of the line like `5 [`, `*[`, `3 `
    prefix = r"^\s*(?:[^A-Za-z]{1,10}\s*)?"

    # STRUCTURAL WORDS (Optional)
    struct = r"(?:[A-Za-z]{2,15}\s+)?(?:[\dIVXLCDM]+\s*\.?\s*)?"

    # SUFFIX SINKHOLE: Eats dashes, dots, spaces, before capturing the run-in body text
    suffix = r"[\s\.\:\-\—\–\_\=\|\*\[\]]*(.*)$"

    pattern = prefix + struct + flex + suffix

    return re.compile(pattern, re.IGNORECASE)

def compile_universal_blueprint(toc_json):
    """
    Safely iterates through the JSON, building the hyper-flexible matching rules.
    """
    compiled_engine = []

    for block in toc_json:
        for identifier, content in block.items():

            if identifier:
                compiled_engine.append({
                    "type": "level_1",
                    "markdown": f"# {identifier}",
                    "raw_text": identifier,
                    "regex": create_universal_regex(identifier)
                })

            chapter_headings = content.get("chapter heading", [])
            if isinstance(chapter_headings, str): chapter_headings = [chapter_headings]
            for heading in chapter_headings:
                if heading:
                    compiled_engine.append({
                        "type": "level_2",
                        "markdown": f"## {heading}",
                        "raw_text": heading,
                        "regex": create_universal_regex(heading),
                        "parent_l1": identifier
                    })

            sections = content.get("section", [])
            if isinstance(sections, str): sections = [sections]
            for section in sections:
                if section:
                    parent_heading = chapter_headings[0] if chapter_headings else None
                    compiled_engine.append({
                        "type": "level_3",
                        "markdown": f"### {section}",
                        "raw_text": section,
                        "regex": create_universal_regex(section),
                        "parent_l1": identifier,
                        "parent_l2": parent_heading
                    })

            sub_sections = content.get("sub section", [])
            if isinstance(sub_sections, str): sub_sections = [sub_sections]
            for sub in sub_sections:
                if sub:
                    compiled_engine.append({
                        "type": "level_4",
                        "markdown": f"#### {sub}",
                        "raw_text": sub,
                        "regex": create_universal_regex(sub)
                    })

    return compiled_engine


# ─────────────────────────────────────────
# State Machine Execution
# ─────────────────────────────────────────

def process_document(raw_text, regex_engine):
    lines = raw_text.split('\n')
    markdown_lines = []

    seen_l1 = set()
    seen_l2 = set()

    for line in lines:
        clean_line = line.strip()

        if not clean_line:
            markdown_lines.append("")
            continue

        matched = False

        for rule in regex_engine:
            if not rule["regex"]:
                continue

            match = rule["regex"].match(clean_line)

            if match:
                # --- STATE MACHINE: Auto-Inject Missing Parents ---
                if rule["type"] == "level_1":
                    seen_l1.add(rule["raw_text"])

                elif rule["type"] == "level_2":
                    parent_l1 = rule.get("parent_l1")
                    if parent_l1 and parent_l1 not in seen_l1:
                        markdown_lines.append(f"# @@H1@@ {parent_l1}")
                        seen_l1.add(parent_l1)
                    seen_l2.add(rule["raw_text"])

                elif rule["type"] == "level_3":
                    parent_l1 = rule.get("parent_l1")
                    parent_l2 = rule.get("parent_l2")
                    if parent_l1 and parent_l1 not in seen_l1:
                        markdown_lines.append(f"# @@H1@@ {parent_l1}")
                        seen_l1.add(parent_l1)
                    if parent_l2 and parent_l2 not in seen_l2:
                        markdown_lines.append(f"## @@H2@@ {parent_l2}")
                        seen_l2.add(parent_l2)

                # # --- 1. INJECT THE PERFECT MARKDOWN HEADER ---
                # # This guarantees "35. Costs." instead of the ugly "*[35. Costs. -( 1 )"
                # markdown_lines.append(rule["markdown"])

                # --- 1. INJECT THE PERFECT MARKDOWN HEADER WITH TROJAN TAG ---
                # Replacing rule["markdown"] to dynamically inject the exact level tag
                if rule["type"] == "level_1":
                    markdown_lines.append(f"# @@H1@@ {rule['raw_text']}")
                elif rule["type"] == "level_2":
                    markdown_lines.append(f"## @@H2@@ {rule['raw_text']}")
                elif rule["type"] == "level_3":
                    markdown_lines.append(f"### @@H3@@ {rule['raw_text']}")
                elif rule["type"] == "level_4":
                    markdown_lines.append(f"#### @@H4@@ {rule['raw_text']}")

                # --- 2. RESCUE THE RUN-IN BODY TEXT ---
                raw_run_in = match.group(1)
                # Strips leftover dashes, spaces, and OCR noise but KEEPS "( 1 )"
                clean_run_in = raw_run_in.lstrip(" .:;,-—–_=|*~\t\n\r")

                if clean_run_in:
                    markdown_lines.append(clean_run_in.strip())

                matched = True
                break

        if not matched:
            if len(clean_line) < 4 and clean_line.isdigit():
                continue
            markdown_lines.append(line)

    return "\n".join(markdown_lines)

# ─────────────────────────────────────────
# Main Processing Function  
# ─────────────────────────────────────────


def process_and_fix_pdf(pdf_path: str, toc_json_data: list):
    print("1. Extracting PDF and Injecting Page Numbers...")
    converter = DocumentConverter()
    initial_result = converter.convert(pdf_path)
    doc = initial_result.document

    citation_verify_pattern = re.compile(
        r"^[\[\(-]*\s*\d+\s*[\]\)-]*\.?\s+.*\b(?:v|vs|AIR|SCC|Cr LJ|SC|Edn|All LJ)\b", 
        re.IGNORECASE | re.DOTALL
    )

    for item, level in doc.iterate_items():
        if isinstance(item, TextItem) and item.text:
            text = item.text.strip()

            # Layer 1: Docling footnote label + Regex Verification
            if item.label == DocItemLabel.FOOTNOTE:
                # Only delete if it actually looks like a legal citation
                if citation_verify_pattern.search(text):
                    item.text = ""
                    continue
                else:
                    # If Docling labeled it a footnote but it lacks citation keywords,
                    # we let it survive and fall through to the page injection below.
                    pass

            # Page injection
            if item.prov and len(item.prov) > 0:
                page_no = item.prov[0].page_no
                if len(text.split()) > 3:
                    item.text = f"{text} @@PAGE_{page_no}@@"

    raw_text = doc.export_to_text()

    synopsis_removed = remove_synopsis_block(raw_text)
    fully_cleaned = remove_ocr_citation_footnotes(synopsis_removed)

    print("2. Fixing Markdown Hierarchy with Step 3 & 4...")
    # Compile the Regex engine from the LLM JSON (Step 3)
    regex_engine = compile_universal_blueprint(toc_json_data)

    # Process the raw markdown through our state machine (Step 4)
    fixed_markdown = process_document(fully_cleaned, regex_engine)

    print("3. Re-ingesting fixed Markdown into Docling...")
    md_bytes = io.BytesIO(fixed_markdown.encode('utf-8'))
    md_stream = DocumentStream(name="fixed_document.md", stream=md_bytes)

    fixed_result = converter.convert(md_stream)
    rich_document = fixed_result.document

    print("4. Applying Hybrid Chunker to the fixed document...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=510, merge_peers=True)

    final_chunks = list(chunker.chunk(dl_doc=rich_document))

    print("5. Extracting injected pages into metadata payloads...")
    database_payloads = []

    for chunk in final_chunks:
        chunk_text = chunk.text

        # Safely get headings if they exist
        raw_headings = chunk.meta.headings if chunk.meta.headings else []

        # Find all injected tags like "[Page 2]"
        page_matches = re.findall(r'@@PAGE_(\d+)@@', chunk_text)
        pages = list(set([int(p) for p in page_matches]))

        # Clean the tags out of the body text
        clean_text = re.sub(r'@@PAGE_(\d+)@@', '', chunk_text).strip()

        # Clean the tags out of the headings
        clean_headings = [re.sub(r'@@PAGE_(\d+)@@', '', h).strip() for h in raw_headings]

        # Skip empty chunks that might just be leftover formatting
        if not clean_text:
            continue

        database_payloads.append({
            "page_content": clean_text,
            "metadata": {
                "source": pdf_path,
                "pages": pages,  # Returns a list of all pages this chunk spans!
                "headings": clean_headings
            }
        })

    return database_payloads

