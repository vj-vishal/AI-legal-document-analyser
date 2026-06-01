from src.legal_rag.rag.engine import RAGPipeline, BM25sDiskRetriever
import src.legal_rag.config as config
import os

def template_embedding():
    rag = RAGPipeline("template_doc", embedding_model= config.EMBEDDING_MODEL)
    rag.process_json(config.PROCESSED_DIR / "template_doc_v1.json", store_docs= False)

def knowledge_base_embedding(json_path: str, _bm25_path: str):
    rag = RAGPipeline("knowledge_base", embedding_model= config.EMBEDDING_MODEL)
    rag.process_json(json_path, store_docs= True)
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

if __name__ == "__main__":
    template_embedding()