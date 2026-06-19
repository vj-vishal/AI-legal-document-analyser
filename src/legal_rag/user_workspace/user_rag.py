from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple
import pickle
import ijson
import bm25s as _bm25s
from dotenv import load_dotenv
from pydantic import ConfigDict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import src.legal_rag.config as config

load_dotenv()  


class BM25sDiskRetriever(BaseRetriever):
    """
    LangChain-compatible BM25 retriever backed by bm25s with full disk persistence.

    Each user gets an isolated index under:
        {config.BM25_INDEX_DIR}/user_{collection_name}/
            ├── *.index          (bm25s scoring model — IDF weights, vocab)
            └── corpus_docs.pkl  (serialized List[Document] with all metadata)

    bm25s.save() stores only the model; the Document list (text + metadata)
    is pickled separately and looked up by the integer indices bm25s returns
    at query time.
    """

    index_path: str
    k: int = config.BM25_K

    _bm25_index: Any = None
    _corpus_docs: List[Document] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def build_and_save(cls, docs: List[Document], index_path: str) -> "BM25sDiskRetriever":
        """
        Build a fresh BM25 index from docs and persist to disk.
        Called after every successful process_json() run.
        BM25 does not support incremental updates — always rebuilt from full corpus.
        """
        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)

        corpus_texts = [doc.page_content for doc in docs]
        corpus_tokens = _bm25s.tokenize(corpus_texts, stopwords="en")

        index = _bm25s.BM25()
        index.index(corpus_tokens)
        index.save(str(path))

        with open(path / "corpus_docs.pkl", "wb") as f:
            pickle.dump(docs, f)

        obj = cls(index_path=index_path)
        obj._bm25_index = index
        obj._corpus_docs = docs
        return obj
    
    @classmethod
    def load_from_disk(cls, index_path: str) -> "BM25sDiskRetriever":
        """
        Load an existing BM25 index from disk.
        Called on server restart — avoids re-embedding already processed documents.
        """
        path = Path(index_path)
        corpus_pkl = path / "corpus_docs.pkl"

        if not corpus_pkl.exists():
            raise FileNotFoundError(
                f"No BM25 index found at '{index_path}'. "
                "The user must upload a document before querying."
            )

        index = _bm25s.BM25.load(str(path))
        with open(corpus_pkl, "rb") as f:
            docs: List[Document] = pickle.load(f)

        obj = cls(index_path=index_path)
        obj._bm25_index = index
        obj._corpus_docs = docs
        return obj

    @property
    def index_exists(self) -> bool:
        """True if a persisted index already exists on disk for this user."""
        return (Path(self.index_path) / "corpus_docs.pkl").exists()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Lazy-load from disk on cache miss or cold start
        if self._bm25_index is None:
            loaded = BM25sDiskRetriever.load_from_disk(self.index_path)
            self._bm25_index = loaded._bm25_index
            self._corpus_docs = loaded._corpus_docs

        if not self._corpus_docs:
            return []

        query_tokens = _bm25s.tokenize([query], stopwords="en")
        k = min(self.k, len(self._corpus_docs))

        # bm25s returns integer indices (shape: n_queries × k) when no corpus
        # was passed to .save(). Map indices back to Document objects.
        results, _ = self._bm25_index.retrieve(query_tokens, k=k)
        indices = results[0].tolist()
        return [self._corpus_docs[i] for i in indices]


class RAGPipeline:

    def __init__(self, collection_name: str = None, embedding_model: str = None, embeddings = None):
        Path(config.VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

        _collection = collection_name or config.COLLECTION_NAME

        # Per-user BM25 index path — mirrors Chroma collection isolation
        self._bm25_path = str(
            Path(config.BM25_INDEX_DIR) / f"user_{_collection}"
        )

        if embeddings is not None:
            self.ef = embeddings
        else:
            if embedding_model is None:
                raise ValueError("Either 'embeddings' or 'embedding_model' must be provided.")
            
            self.ef = HuggingFaceEmbeddings(
                model_name=embedding_model               
            )

        self.vector_store = Chroma(
            collection_name=_collection,
            embedding_function=self.ef,
            persist_directory=str(config.VECTORSTORE_DIR)
        )

        self.lc_docs: List[Document] = []
            

    def process_json(self, json_data: List[Dict], store_docs: bool = False) -> None:
        # self.vector_store.reset_collection()
        self.lc_docs.clear()

        batch_docs = []
        batch_ids = []

        # with open(json_path, "rb") as f:
        for entry in json_data:
            meta = entry.get("metadata", {})

            # meta = {
            #     "id":               entry.get("chunk_id") or "",
            #     "title":            entry.get("structural_path") or "",
            #     "source":           raw_meta.get("source_file") or "",
            #     "page":             raw_meta.get("page") or 0
            # }

            doc = Document(page_content=entry["page_content"], metadata=meta)

            batch_docs.append(doc)
            batch_ids.append(str(uuid4()))

            if store_docs:
                self.lc_docs.append(doc)

            if len(batch_docs) >= config.BATCH_SIZE:
                self.vector_store.add_documents(batch_docs, ids=batch_ids)
                batch_docs.clear()
                batch_ids.clear()

        if batch_docs:
            self.vector_store.add_documents(batch_docs, ids=batch_ids)

    # def _build_bm25(self) -> None:
    #     """
    #     Build a fresh BM25 index from self.lc_docs, persist it to the
    #     user's dedicated disk folder, then assemble the QA chain.
    #     Called by the API/router after process_json() completes.
    #     """
    #     bm25 = BM25sDiskRetriever.build_and_save(
    #         docs=self.lc_docs,
    #         index_path=self._bm25_path
    #     )



if __name__ == "__main__":
    pipeline= RAGPipeline()
    pipeline.process_json(config.CHUNK_JSON)
    pipeline._build_bm25()
