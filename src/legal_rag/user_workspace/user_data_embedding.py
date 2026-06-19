from src.legal_rag.user_workspace.user_rag import RAGPipeline, BM25sDiskRetriever
import src.legal_rag.config as config
import os
from typing import List, Dict
from src.legal_rag.user_workspace.data_extraction_docling import parse_document, chunk_to_langchain_documents

def user_knowledge_base_embedding(collection_name: str, json_data: List[Dict], _bm25_path: str):
    rag = RAGPipeline(collection_name, embedding_model= "microsoft/harrier-oss-v1-0.6b")
    rag.process_json(json_data, store_docs= True)
    if os.path.exists(_bm25_path):
        # Load old docs and combine with new
        old_retriever = BM25sDiskRetriever.load_from_disk(_bm25_path)
        all_docs = old_retriever._corpus_docs + rag.lc_docs
        print(f"Existing index found — merging {len(old_retriever._corpus_docs)} old + {len(rag.lc_docs)} new docs")
    else:
        # Fresh build
        all_docs = rag.lc_docs
        print(f"No existing index — fresh build with {len(rag.lc_docs)} docs")
    BM25sDiskRetriever.build_and_save(
            docs=all_docs,
            index_path=_bm25_path
        )
    
def orchestrator(pdf_path: str, collection_name: str, kb_document_id: str, kb_id: str):
    # pdf_path = "data/intermediate/batches/batch_31_to_40.pdf"
    parse_doc= parse_document(pdf_path)
    json_data = chunk_to_langchain_documents(parse_doc, pdf_path, kb_document_id, kb_id)
    # collection_name = "user_123_kb"
    _bm25_path = str(config.BM25_INDEX_DIR / collection_name)
    user_knowledge_base_embedding(collection_name, json_data, _bm25_path)
    print("Orchestration complete.")

if __name__ == "__main__":
    # Example usage
    orchestrator("data/intermediate/batches/batch_31_to_40.pdf", "user_123_kb", "550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001")