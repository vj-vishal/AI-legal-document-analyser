from src.legal_rag.user_workspace.user_rag import RAGPipeline, BM25sDiskRetriever
import src.legal_rag.config as config
import os
from typing import List, Dict
from src.legal_rag.user_workspace.data_extraction_docling import parse_document, chunk_to_langchain_documents

# 1. Import trace from OpenTelemetry
from opentelemetry import trace

# 2. Initialize the tracer for this module
tracer = trace.get_tracer("legal_rag_worker.orchestrator")

def user_knowledge_base_embedding(collection_name: str, json_data: List[Dict], _bm25_path: str):

    # Trace the Dense Vector Embedding processing
    with tracer.start_as_current_span("rag_process_json") as rag_span:
        rag_span.set_attribute("app.collection_name", collection_name)
        rag_span.set_attribute("app.input_chunks_count", len(json_data))

        rag = RAGPipeline(collection_name, embedding_model= "microsoft/harrier-oss-v1-0.6b")
        rag.process_json(json_data, store_docs= True)

        # Record how many Langchain docs were actually created
        rag_span.set_attribute("app.lc_docs_created", len(rag.lc_docs))

    # Trace the Sparse BM25 Index merging and saving
    with tracer.start_as_current_span("bm25_index_build") as bm25_span:
        bm25_span.set_attribute("app.bm25_path", _bm25_path)
        if os.path.exists(_bm25_path):
            # Load old docs and combine with new
            old_retriever = BM25sDiskRetriever.load_from_disk(_bm25_path)
            all_docs = old_retriever._corpus_docs + rag.lc_docs
            print(f"Existing index found — merging {len(old_retriever._corpus_docs)} old + {len(rag.lc_docs)} new docs")

            # Tag the span so you know it was a merge
            bm25_span.set_attribute("app.index_status", "merged")
            bm25_span.set_attribute("app.old_docs_count", len(old_retriever._corpus_docs))

        else:
            # Fresh build
            all_docs = rag.lc_docs
            print(f"No existing index — fresh build with {len(rag.lc_docs)} docs")

            # Tag the span so you know it was a fresh build
            bm25_span.set_attribute("app.index_status", "fresh_build")

        bm25_span.set_attribute("app.total_docs_indexed", len(all_docs))

        BM25sDiskRetriever.build_and_save(
                docs=all_docs,
                index_path=_bm25_path
            )
    
def orchestrator(pdf_path: str, collection_name: str, kb_document_id: str, kb_id: str, user_id: str):
    
    # Trace Document Parsing
    with tracer.start_as_current_span("parse_document") as parse_span:
        parse_span.set_attribute("app.pdf_path", pdf_path)
        parse_span.set_attribute("app.user_id", user_id)
        parse_span.set_attribute("app.kb_id", kb_id)

        parse_doc= parse_document(pdf_path)

        # Trace Chunking
    with tracer.start_as_current_span("chunk_to_langchain_documents") as chunk_span:
        chunk_span.set_attribute("app.kb_document_id", kb_document_id)

        json_data = chunk_to_langchain_documents(parse_doc, pdf_path, kb_document_id, kb_id, user_id)

        # Safely record chunk count if json_data is a list
        if isinstance(json_data, list):
            chunk_span.set_attribute("app.chunk_count", len(json_data))
    
    with tracer.start_as_current_span("user_knowledge_base_embedding") as embed_span:
        _bm25_path = str(config.BM25_INDEX_DIR / collection_name)
        user_knowledge_base_embedding(collection_name, json_data, _bm25_path)
        
    print("Orchestration complete.")

if __name__ == "__main__":
    # Example usage
    orchestrator("data/intermediate/batches/batch_31_to_40.pdf", "user_123_kb", "550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002")