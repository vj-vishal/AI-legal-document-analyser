from typing import List
import tiktoken

def format_data(chunks: List):
    formatted_chunks = []
    # Define keys that are purely for your Python backend and should NEVER go to the LLM
    METADATA_BLACKLIST = {"rerank_score", "score", "collection_name", "vector_id"}

    for i, chunk in enumerate(chunks):
        # Start building the metadata block dynamically
        metadata_lines = []
        
        # 1. Run the loop for metadata items
        for key, value in chunk.metadata.items():
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
        {chunk.page_content.strip()}
      </content>
    </chunk>"""

        formatted_chunks.append(formatted_data)
    
    return formatted_chunks

def count_tokens_locally(text: str, model_name: str = "gpt-4o") -> int:
    """
    Counts the number of tokens in a text string locally without calling an API.
    """
    # 1. Get the encoding specific to the model
    # (gpt-4o and gpt-4 use the 'o200k_base' or 'cl100k_base' encodings)
    encoding = tiktoken.encoding_for_model(model_name)
    
    # 2. Encode the text into a list of integers (token IDs)
    token_list = encoding.encode(text)
    
    # 3. Return the total count
    return len(token_list)
