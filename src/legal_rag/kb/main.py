from src.legal_rag.kb.router import router
from src.legal_rag.rag.engine import RAGPipeline, BM25sDiskRetriever, HybridRerankRetriever
import src.legal_rag.config as config
from src.legal_rag.kb.api_integration import IndianKanoonClient, clean_fragment, retrieve_from_kanoon
from langchain_huggingface import HuggingFaceEmbeddings
from src.legal_rag.kb.llm_as_judge import RetrievalDecisionManager, GroqRetrievalJudge


class KnowledgeBaseManager:
    """
    Manages all retrieval pipelines for the legal document analyzer.
    All heavy resources (embedding models, BM25 indexes) are loaded
    once at initialization and reused for every query.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="microsoft/harrier-oss-v1-0.6b"
        )

        # Dense pipelines
        self.kb_pipeline = RAGPipeline(
            collection_name="knowledge_base",
            embeddings=self.embeddings
        )
        self.temp_pipeline = RAGPipeline(
            collection_name="template_doc",
            embeddings=self.embeddings
        )
        self.user_pipeline = RAGPipeline(
            collection_name="0d3493be-c6b9-47dd-9387-7301b812b52a",
            embeddings=self.embeddings
        )

        # BM25 sparse retriever(internal documents)
        bm25_kb = BM25sDiskRetriever.load_from_disk(
            str(config.BM25_INDEX_DIR / "knowledge_base")
        )
        bm25_kb.k = config.BM25_K

        # BM25 sparse retriever(users documents)
        bm25_user = BM25sDiskRetriever.load_from_disk(
            str(config.BM25_INDEX_DIR / "0d3493be-c6b9-47dd-9387-7301b812b52a")
        )
        bm25_user.k = config.BM25_K

        # Hybrid retriever (BM25 + dense + rerank)
        self.kb_hybrid = HybridRerankRetriever.from_components(
            bm25_internal=bm25_kb,
            bm25_user=bm25_user,
            internal_store=self.kb_pipeline.vector_store,
            user_store= self.user_pipeline.vector_store
        )

    # ─── Private Retrievers ────────────────────────────────────────────────

    def _retrieve_from_kb(self, query: str, user_id: str = None, kb_id: str = None, kb_document_id: str = None):
        return self.kb_hybrid.invoke(query, user_id=user_id, kb_id=kb_id, kb_document_id=kb_document_id)

    def _retrieve_from_template(self, query: str):
        dense = self.temp_pipeline.vector_store.as_retriever(
            search_kwargs={"k": 1}
        )
        return dense.invoke(query)

    def _retrieve_with_fallback(self, query: str, user_id: str , kb_id: str , kb_document_id: str):
        chunks = self._retrieve_from_kb(query, user_id=user_id, kb_id=kb_id, kb_document_id=kb_document_id)
        # top_score = chunks[0].metadata.get("rerank_score", 0) if chunks else 0
        retrieval_judge = GroqRetrievalJudge()
        decision_manager = RetrievalDecisionManager()

        result = retrieval_judge.judge(query, chunks)
        final_result = decision_manager.evaluate(result)
        if final_result.decision == "PASS":
            final_chunks= decision_manager.collect_final_chunks(chunks, final_result)
            
        elif final_result.decision == "FALLBACK":
            client = IndianKanoonClient()
            final_chunks= retrieve_from_kanoon(query, client)

        return final_chunks

    # ─── Public API ───────────────────────────────────────────────────────

    def route_and_retrieve(self, query: str, user_id: str, kb_id: str , kb_document_id: str):
        route = router(query).name

        if route == "knowledge_base":
            return self._retrieve_with_fallback(query, user_id=user_id, kb_id=kb_id, kb_document_id=kb_document_id)
        elif route == "template_doc":
            return self._retrieve_from_template(query)
        else:
            return []


if __name__ == "__main__":
    query = """Under Section 60, which properties of a judgment-debtor are exempt from attachment and sale during execution of a decree?"""
    pipeline = KnowledgeBaseManager()
    results = pipeline.route_and_retrieve(query, user_id= "6544c0d7-aa3c-4dc9-b0aa-faed878d7ff3", kb_id= "0d3493be-c6b9-47dd-9387-7301b812b52a", kb_document_id= "3ce373ed-0ab6-4f7a-b55b-720e2a9445a6")
    print(results)
    # results= pipeline.route_and_retrieve(query)
    # print(results)
    # print(50*"==")
    # for r in results:     
    #     print(r.page_content)
    #     print(r.metadata)
    #     print(50 * "=")
    # query = "Give me Non rent agreement template"
    # results = routing(query)
    # for r in results:
    #     print(r.page_content)
    #     print(r.metadata)
    #     print(50 * "=")   
    
    # query = "Within what period can an applicant appeal a decision to reject their legal aid application?"
    # results = template_retriever(query)
    # knowledge_base_retriever(query, str(config.BM25_INDEX_DIR / "knowledge_base"))
    # results = knowledge_base_retriever(query, str(config.BM25_INDEX_DIR / "knowledge_base")) 
    # print(results)
    # for r in results:
    #     print(r.page_content)
    #     print(r.metadata)
    #     print(50 * "=")   