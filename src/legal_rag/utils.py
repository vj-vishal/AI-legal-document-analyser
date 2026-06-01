from typing import List

def format_data(chunks: List):
    formatted_chunks = []
    # Define keys that are purely for your Python backend and should NEVER go to the LLM
    METADATA_BLACKLIST = {"rerank_score", "score", "collection_name", "vector_id"}

    for i, chunk in enumerate(chunks):
        # Start building the metadata block dynamically
        metadata_lines = []
        
        # 1. Run the loop for metadata items
        for key, value in chunk["metadata"].items():
            # Skip internal backend keys
            if key in METADATA_BLACKLIST or value is None:
                continue
                
            # Format the key to look clean (e.g., 'chapter_title' -> 'Chapter Title')
            clean_key = key.replace("_", " ").title()
            metadata_lines.append(f"    {clean_key}: {value}")
        
        # Join all valid metadata lines together
        metadata_string = "\n".join(metadata_lines)
        
        # 2. Construct the clean XML chunk
        formatted_data = f"""<chunk id="{i+1}">
      <source_metadata>
    {metadata_string}
      </source_metadata>
      <content>
        {chunk["page_content"].strip()}
      </content>
    </chunk>"""

        formatted_chunks.append(formatted_data)
    
    return formatted_chunks
