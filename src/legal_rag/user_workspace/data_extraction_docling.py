import os
import json
import time
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

# Docling Imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

# LangChain Imports
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Pydantic
from pydantic import BaseModel, Field
import src.legal_rag.config as config
from src.legal_rag.user_workspace.key_value_extraction import _clean_json, extract_data_from_raw_text

# Load environment variables
load_dotenv()

# ==========================================
# 2. PIPELINE FUNCTIONS
# ==========================================

def parse_document(pdf_path: str) -> Any:
    """Parses the PDF layout using Docling Vision Parser."""
    print("1. Initializing Docling Vision Parser...")
    converter = DocumentConverter()

    print(f"2. Analyzing Layout of {pdf_path} (This takes a few seconds)...")
    docling_result = converter.convert(pdf_path)
    return docling_result.document


def chunk_to_langchain_documents(rich_document: Any, pdf_path: str) -> List[Document]:
    """Applies Hybrid Chunking and converts to standard LangChain Documents."""
    print("3. Applying Hybrid Chunking...")
    
    # Initialize tokenizer and chunker
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    chunker = HybridChunker(
        tokenizer=tokenizer, 
        max_tokens=510, # Keeps chunks perfectly sized for the LLM
        merge_peers=True
    )

    docling_chunks = list(chunker.chunk(dl_doc=rich_document))

    documents = []

    # Bridge: Convert Docling chunks into standard LangChain Document objects
    for i, chunk in enumerate(docling_chunks):
        headings = chunk.meta.headings if chunk.meta.headings else ["Root"]
        heading_path = " > ".join(headings)

        doc= {
            "chunk_id": i,
            "source_file": pdf_path,
            "structural_path": heading_path,
            "page_number": chunk.meta.doc_items[0].prov[0].page_no, #if chunk.meta.doc_items else None,
            "text": chunk.text
        }
        documents.append(doc)
    return documents

        
    #     lc_doc = Document(
    #         page_content=chunk.text,
    #         metadata={
    #             "chunk_id": i,
    #             "source_file": pdf_path,
    #             "structural_path": heading_path,
    #             "page_number": chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else None
    #         }
    #     )
    #     langchain_documents.append(lc_doc)

    # print(f"   -> Created {len(langchain_documents)} structural chunks.\n")
    # return langchain_documents

if __name__ == "__main__":
    parse_doc= parse_document(config.BATCH_DIR/"batch_31_to_40.pdf")
    chunks = chunk_to_langchain_documents(parse_doc, config.BATCH_DIR/"batch_31_to_40.pdf")

    print(chunks)

    # with open(config.CHUNK_JSON, "w", encoding="utf-8") as f:
    #     json.dump(chunks, f, indent=2)

    # raw_text= parse_doc.export_to_text()
    # extracted_data = extract_data_from_raw_text(raw_text)

    # with open(config.KEY_DATA_JSON, "w", encoding="utf-8") as f:
    #     json.dump(extracted_data, f, indent=2)
