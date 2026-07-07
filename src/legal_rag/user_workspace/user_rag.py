from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple,Set, Optional
import numpy as np
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
    LangChain-compatible BM25 retriever backed by bm25s with full disk persistence
    and O(1) metadata pre-filtering.
    """

    index_path: str
    k: int = 3

    _bm25_index: Any = None
    _corpus_docs: List[Document] = []
    
    # ── Added: In-Memory Metadata Indices ──
    _user_idx: dict[str, Set[int]] = {}
    _kb_idx: dict[str, Set[int]] = {}
    _doc_idx: dict[str, Set[int]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build_metadata_indices(self) -> None:
        """Build O(1) lookups for metadata for lightning-fast pre-filtering."""
        self._user_idx, self._kb_idx, self._doc_idx = {}, {}, {}
        for idx, doc in enumerate(self._corpus_docs):
            u_id = doc.metadata.get("user_id")
            k_id = doc.metadata.get("kb_id")
            d_id = doc.metadata.get("kb_document_id")

            if u_id: self._user_idx.setdefault(u_id, set()).add(idx)
            if k_id: self._kb_idx.setdefault(k_id, set()).add(idx)
            if d_id: self._doc_idx.setdefault(d_id, set()).add(idx)

    @classmethod
    def build_and_save(cls, docs: List[Document], index_path: str) -> "BM25sDiskRetriever":
        """
        Build a fresh BM25 index from docs and persist to disk.
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
        obj._build_metadata_indices()  # Initialize the fast-lookups
        return obj

    @classmethod
    def load_from_disk(cls, index_path: str) -> "BM25sDiskRetriever":
        """
        Load an existing BM25 index from disk.
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
        obj._build_metadata_indices()  # Initialize the fast-lookups
        return obj

    @property
    def index_exists(self) -> bool:
        """True if a persisted index already exists on disk for this user."""
        return (Path(self.index_path) / "corpus_docs.pkl").exists()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        user_id: Optional[str] = None,
        kb_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Document]:
        
        # Lazy-load from disk on cache miss or cold start
        if self._bm25_index is None:
            loaded = BM25sDiskRetriever.load_from_disk(self.index_path)
            self._bm25_index = loaded._bm25_index
            self._corpus_docs = loaded._corpus_docs
            # Copy over the in-memory indices so we don't have to rebuild them
            self._user_idx = loaded._user_idx
            self._kb_idx = loaded._kb_idx
            self._doc_idx = loaded._doc_idx

        if not self._corpus_docs:
            return []

        query_tokens_obj = _bm25s.tokenize([query], stopwords="en")
        filters_active = any([user_id, kb_id, document_id])

        # ── Step 1: Unfiltered Path (Fast Default) ─────────────────────────
        if not filters_active:
            k = min(self.k, len(self._corpus_docs))
            results, _ = self._bm25_index.retrieve(query_tokens_obj, k=k)
            indices = results[0].tolist()
            return [self._corpus_docs[i] for i in indices]

        # ── Step 2: Instant Pre-Filter via Set Intersection ────────────────
        sets_to_intersect = []
        if user_id: sets_to_intersect.append(self._user_idx.get(user_id, set()))
        if kb_id: sets_to_intersect.append(self._kb_idx.get(kb_id, set()))
        if document_id: sets_to_intersect.append(self._doc_idx.get(document_id, set()))

        allowed_set = set.intersection(*sets_to_intersect)

        if not allowed_set:
            return []

        # ── Step 3: Fetch Raw Scores and Sort ONLY the Allowed Docs ────────
        allowed_list = list(allowed_set)

        try:
            # Check if it's a tuple (older bm25s version) or an object (newer)
            if isinstance(query_tokens_obj, tuple):
                ids_matrix, vocab_dict = query_tokens_obj
            else:
                ids_matrix = query_tokens_obj.ids
                vocab_dict = query_tokens_obj.vocab
                
            # Invert the dictionary so ID maps to String
            inv_vocab = {v: k for k, v in vocab_dict.items()}
            
            # Translate the integer array for our query back to a list of strings
            string_tokens = [inv_vocab[token_id] for token_id in ids_matrix[0]]
            
        except Exception:
            # Bulletproof fallback if the library behavior changes unexpectedly
            import re
            string_tokens = re.findall(r'(?u)\b\w\w+\b', query.lower())
            print("Warning: Failed to decode query tokens. Falling back to raw query string.")

        # get_scores() returns the raw numpy array of scores for all documents without sorting
        raw_scores = self._bm25_index.get_scores(string_tokens)
        
        # Depending on bm25s version, scores might be 2D. Flatten it safely.
        if len(raw_scores.shape) > 1:
            raw_scores = raw_scores.flatten()

        # Extract the scores for just our filtered subset using NumPy indexing
        filtered_scores = raw_scores[allowed_list]

        # Sort only the filtered subset to find the top-K
        # np.argsort sorts ascending, so we take the last 'k' elements and reverse them [::-1]
        k = min(self.k, len(filtered_scores))
        top_k_relative_indices = np.argsort(filtered_scores)[-k:][::-1]

        # Map back to the original document list
        return [
            self._corpus_docs[allowed_list[rel_idx]]
            for rel_idx in top_k_relative_indices
        ]



class RAGPipeline:

    def __init__(self, collection_name: str = None, embedding_model: str = None, embeddings = None):
        Path(config.VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

        _collection = collection_name or config.COLLECTION_NAME

        # Per-user BM25 index path — mirrors Chroma collection isolation
        self._bm25_path = str(
            Path(config.BM25_INDEX_DIR) / f"{_collection}"
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

