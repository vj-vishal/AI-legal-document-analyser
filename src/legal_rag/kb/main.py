from  dotenv import load_dotenv
import os
load_dotenv()
from src.legal_rag.kb.router import router
from src.legal_rag.rag.engine import RAGPipeline, BM25sDiskRetriever, HybridRerankRetriever
import src.legal_rag.config as config
from src.legal_rag.kb.api_integration import IndianKanoonClient, clean_fragment, retrieve_from_kanoon
from langchain_huggingface import HuggingFaceEmbeddings
from src.legal_rag.kb.llm_as_judge import RetrievalDecisionManager, GroqRetrievalJudge
import torch
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor

user_kb_id = os.getenv("USER_KB_ID")

# FORCE offline mode to prevent 5-second HuggingFace network pings
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. Import trace from OpenTelemetry
from opentelemetry import trace

# 2. Initialize the tracer for this module
tracer = trace.get_tracer("legal_rag_api.retrieval_orchestrator")


class KnowledgeBaseManager:
    """
    Manages all retrieval pipelines for the legal document analyzer.
    All heavy resources (embedding models, BM25 indexes) are loaded
    once at initialization and reused for every query.
    """

    def __init__(self):

        # with tracer.start_as_current_span("loading_embd_model") as loading_span:
        #     self.embeddings = HuggingFaceEmbeddings(
        #         model_name="microsoft/harrier-oss-v1-0.6b"
        #     )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        def _load_embeddings():
            return HuggingFaceEmbeddings(
                model_name="microsoft/harrier-oss-v1-0.6b",
                model_kwargs={"device": device, 
                              "model_kwargs": {"torch_dtype": dtype} 
                    }
            )

        def _load_reranker():
            return CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                device=device,
                max_length=1024,
                default_activation_function=None,
                model_kwargs={"torch_dtype": dtype},
            )

        # Execute loading concurrently
        with tracer.start_as_current_span("concurrent_model_loading"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_embd = executor.submit(_load_embeddings)
                future_reranker = executor.submit(_load_reranker)

                self.embeddings = future_embd.result()
                loaded_reranker = future_reranker.result()

        with tracer.start_as_current_span("kb_pipeline") as kb_pipeline_span:
            # Dense pipelines
            self.kb_pipeline = RAGPipeline(
                collection_name="knowledge_base",
                embeddings=self.embeddings
            )

        with tracer.start_as_current_span("temp_pipeline") as temp_pipeline_span:
            self.temp_pipeline = RAGPipeline(
                collection_name="template_doc",
                embeddings=self.embeddings
            )

        with tracer.start_as_current_span("user_kb_pipeline") as user_kb_pipeline_span:
            self.user_pipeline = RAGPipeline(
                collection_name=user_kb_id,
                embeddings=self.embeddings
            )

        with tracer.start_as_current_span("load_internal_bm25") as internal_bm25_span:
            # BM25 sparse retriever(internal documents)
            bm25_kb = BM25sDiskRetriever.load_from_disk(
                str(config.BM25_INDEX_DIR / "knowledge_base")
            )
            bm25_kb.k = config.BM25_K

        with tracer.start_as_current_span("load_user_bm25") as user_bm25_span:
            # BM25 sparse retriever(users documents)
            bm25_user = BM25sDiskRetriever.load_from_disk(
                str(config.BM25_INDEX_DIR / user_kb_id)
            )
            bm25_user.k = config.BM25_K

        with tracer.start_as_current_span("load_hybrid_rerank") as hybrid_rerank_span:
        # Hybrid retriever (BM25 + dense + rerank)
            self.kb_hybrid = HybridRerankRetriever.from_components(
                bm25_internal=bm25_kb,
                bm25_user=bm25_user,
                internal_store=self.kb_pipeline.vector_store,
                user_store= self.user_pipeline.vector_store,
                load_reranker=loaded_reranker

            )

    # ─── Private Retrievers ────────────────────────────────────────────────

    def _retrieve_from_kb(self, query: str, user_id: str = None, kb_id: str = None, kb_document_id: str = None):
        return self.kb_hybrid.invoke(query, user_id=user_id, kb_id=kb_id, kb_document_id=kb_document_id)

    def _retrieve_from_template(self, query: str):
        with tracer.start_as_current_span("template_retrieval") as template_span:
            template_span.set_attribute("app.query", query)

            dense = self.temp_pipeline.vector_store.as_retriever(
            search_kwargs={"k": 1}
        )
        return dense.invoke(query)

    def _retrieve_with_fallback(self, query: str, user_id: str , kb_id: str , kb_document_id: str):
        with tracer.start_as_current_span("fallback") as fallback_span:
            fallback_span.set_attribute("app.user_id", user_id)
            fallback_span.set_attribute("app.kb_id", kb_id)
            fallback_span.set_attribute("app.kb_document_id", kb_document_id)

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

        with tracer.start_as_current_span("route") as route_span:
            route = router(query).name

        with tracer.start_as_current_span("route_to_kb") as routekb_span:
            routekb_span.set_attribute("app.user_id", user_id)
            routekb_span.set_attribute("app.kb_id", kb_id)
            routekb_span.set_attribute("app.kb_document_id", kb_document_id)
            if route == "knowledge_base":
                return self._retrieve_with_fallback(query, user_id=user_id, kb_id=kb_id, kb_document_id=kb_document_id)
            elif route == "template_doc":
                return self._retrieve_from_template(query)
            else:
                return []

    