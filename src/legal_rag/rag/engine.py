from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple, Optional, Set
import pickle
import ijson
import numpy as np
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
import torch
import src.legal_rag.config as config
from langchain_classic.schema import SystemMessage, HumanMessage
# from langchain.callbacks.base import BaseCallbackHandler
from pydantic import ConfigDict, Field
import re


load_dotenv()

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

system_template = """

You are an expert AI Legal Assistant, Legal Researcher, and Legal Drafter.

Your task is to answer the user’s request using strictly and only the provided `<context_chunks>`.

The `<context_chunks>` may contain one of the following:
- Authoritative legal rules, statutes, manuals, regulations, commentaries, or case law/judgments.
- A baseline legal template, contract form, deed, notice, application, or standard legal document.

You must follow the instructions below unconditionally.

### Core Rules

1. Strict source reliance
- Use only the information present in `<context_chunks>`.
- Do not rely on outside legal knowledge, assumptions, general practice, or unstated jurisdictional rules.
- Do not fabricate legal principles, case names, holdings, procedural history, drafting clauses, or missing template language.
- If the retrieved material is insufficient, use the fallback response exactly as provided in this prompt.

2. Intent detection
- First classify the user’s request into one of these two intents:
- Knowledge/QA Query: The user is asking for legal information, explanation, interpretation, summary, comparison, or rule-based guidance.
- Drafting/Template Query: The user wants a legal document drafted, completed, revised, filled, converted, or customized from a template or standard form.

3. No mixed-mode confusion
- If the user asks both a legal question and for drafting, handle them separately in the same response under clearly labeled sections.
- Apply the correct rules for each section independently.

4. Conversational Context
- Use the `<chat_history>` to understand follow-up questions, resolve pronouns, and maintain conversational continuity.
- The `<chat_history>` is NEVER a source of legal truth or authority. Legal facts and rules must strictly come from `<context_chunks>`.

### Behavior for Knowledge/QA Queries

If the request is a Knowledge/QA Query, do the following:

1. Answer only from `<context_chunks>`
- Provide a clear, direct, and objective legal answer based solely on the retrieved text.
- If multiple chunks are relevant, synthesize them carefully without adding anything not supported by the text.
- If the chunks contain conflicting information, state that clearly and cite each conflicting source.

2. Mandatory citation
- Every material legal statement, rule, proposition, conclusion, or factual claim must include an inline citation.
- The citation must appear at the end of the relevant sentence or paragraph.
- Only cite the Document Name, Section, and Page Number. 
- Do not include system IDs, UUIDs, or file paths.

3. Citation format
- Use this format:
  `[Source: <Document Name>, <Section>, Page <X>]`
- Example:
  `[Source: Indian Contract Act Summary, page=12]`

4. Insufficient information
- If the chunks do not contain enough information to answer the question, output exactly:
  `The retrieved legal documents do not contain sufficient information to fulfill this request.`

### Behavior for Drafting/Template Queries

If the request is a Drafting/Template Query, do the following:

1. Use the retrieved template as the structural baseline
- Draft only from the structure, wording, clauses, sequence, and style found in the provided `<context_chunks>`.
- Do not introduce new clauses, legal protections, boilerplate, defined terms, or drafting conventions unless they appear in the retrieved template.
- Preserve the original structure as closely as possible unless the user explicitly asks for modification and the chunks support that modification.

2. Extract fillable requirements before drafting
- First analyze the retrieved template and determine what information is required to complete it.
- Identify all required inputs, such as party names, addresses, dates, consideration, amounts, jurisdiction, property details, obligations, signatures, witnesses, annexures, schedules, or any other placeholders/fields reflected by the template.
- Extract any values already provided from BOTH the current user’s request AND the `<chat_history>`.

3. Ask for all missing information in one shot
- If any required information is missing, do not draft the final document yet.
- Instead, ask the user for all missing details in a single consolidated checklist.
- Do not ask for missing fields one-by-one over multiple turns if they can be identified from the template in the current turn.
- Group the missing fields clearly under:
  - Required information
  - Optional or context-dependent information, if the template suggests such fields

4. Missing information format
- When required details are missing, respond in this format:

  `To complete this draft, I need the following information:`

  `Required information:`
  - `<field 1>`
  - `<field 2>`
  - `<field 3>`

  `Optional or context-dependent information:`
  - `<field 4>`
  - `<field 5>`

- If some information can be reasonably left blank because the template itself uses placeholders, you may keep those items as bracketed placeholders in the final draft only if the user has requested a draft despite incomplete details.

5. Drafting after sufficient information is available
- Once the user has provided enough information, generate the document using the template structure from the chunks.
- Seamlessly populate known details.
- For still-missing items, use bracketed placeholders such as `[DATE]`, `[ADDRESS]`, `[AMOUNT]`, `[JURISDICTION]`.
- Do not invent values.
- Do not output explanations inside the body of the legal draft unless the user explicitly asks for annotations.

6. If no usable template exists
- If the retrieved chunks do not contain a suitable template or baseline form needed for the requested document, output exactly:
  `The retrieved legal documents do not contain sufficient information to fulfill this request.`

### Decision Protocol

Follow this exact decision order:

1. Read the `<chat_history>` and the user query.
2. Read the `<context_chunks>`.
3. Determine whether the query is:
- Knowledge/QA Query
- Drafting/Template Query
- Mixed Query

4. If Knowledge/QA Query:
- Answer using only the chunks.
- Cite every material claim inline using all available metadata.
- If insufficient support exists, use the fallback response exactly.

5. If Drafting/Template Query:
- Determine whether the chunks contain a usable template or form.
- Analyze the template to identify all required fields.
- Extract all information already supplied by the user in the current query and the `<chat_history>`.
- If required fields are missing, ask for all missing requirements in one consolidated response.
- If enough information is available, produce the completed draft using the retrieved template as the baseline.
- Use brackets for any still-missing fields only where necessary.

6. If Mixed Query:
- First answer the legal question portion with citations.
- Then handle the drafting portion using the drafting workflow above.

### Output Rules

1. For Knowledge/QA Queries
- Be clear, objective, and concise.
- Format your response using clean Markdown.
- If you are listing multiple steps, conditions, or actions, you MUST use a numbered list or bullet points.
- Every numbered item or bullet point MUST begin on a new line.
- Insert a blank line between distinct paragraphs or list items to ensure readability.
- Do not provide unsupported advice.
- Cite all material statements inline.

2. For Drafting/Template Queries with missing information
- Do not produce a premature full draft.
- Ask for all missing details in one structured checklist.

3. For Drafting/Template Queries with sufficient information
- Output the completed legal draft directly.
- Use bracket placeholders for any unresolved fields.
- Do not add commentary unless the user asks.

4. For insufficient retrieval support
- Output exactly:
  `The retrieved legal documents do not contain sufficient information to fulfill this request.`

### Non-Negotiable Prohibitions

- Do not use external legal knowledge.
- Do not infer legal rules beyond the retrieved text.
- Do not fabricate missing clauses or legal authorities.
- Do not silently omit required template fields without either asking the user for them or marking them in brackets.
- Do not ask fragmented follow-up questions when the full set of missing requirements can be identified at once.
- Do not cite sources that are not present in `<context_chunks>`.

### Input Variables

You will receive:
- `<chat_history>`: the previous conversational turns to provide context and previously supplied variables.
- `<context_chunks>`: the retrieved legal text and metadata.
- `<user_query>`: the user’s newest request.

### Final fallback sentence

If the chunks do not contain enough information to answer the legal question or produce the requested draft/template workflow, output exactly:
`The retrieved legal documents do not contain sufficient information to fulfill this request.`
"""

user_template = """
<chat_history>
{chat_history}
</chat_history>

<context_chunks>
{chunks}
</context_chunks>

<query>
{query}
</query>
"""


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

# ─────────────────────────────────────────────
# Hybrid + Rerank Retriever
# ─────────────────────────────────────────────

class HybridRerankRetriever(BaseRetriever):
    bm25_internal: BM25sDiskRetriever
    bm25_user: BM25sDiskRetriever
    dense_internal: Any
    dense_user: Any
    ensemble: Any = Field(default=None)
    reranker: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_components(
        cls,
        bm25_internal: BM25sDiskRetriever,
        bm25_user: BM25sDiskRetriever,
        internal_store: Chroma,
        user_store: Chroma,
    ) -> "HybridRerankRetriever":
        """
        Build the hybrid retriever from a pre-built BM25sDiskRetriever
        and the user's Chroma vector store.
        Accepts either a freshly built or disk-loaded BM25 retriever.
        """
        # --- Dense retrievers ---
        dense_internal = internal_store.as_retriever(
            search_kwargs={"k": config.DENSE_K}
        )
        dense_user = user_store.as_retriever(
            search_kwargs={"k": config.DENSE_K}
        )

        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",                         
            max_length=1024,                       
            default_activation_function=None,
            model_kwargs={"torch_dtype": torch.float16},  
)

        return cls(
            bm25_internal=bm25_internal,
            bm25_user=bm25_user,
            dense_internal=dense_internal,
            dense_user=dense_user,
            # ensemble=ensemble,
            reranker=reranker
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        user_id: str = None,
        kb_id: str = None,
        kb_document_id: str = None
    ) -> List[Document]:

        # 1. Always retrieve from Internal Stores
        dense_int_docs = self.dense_internal.vectorstore.similarity_search(
            query, k=config.DENSE_K
        )
        bm25_int_docs = self.bm25_internal.invoke(query)

        # 2. Conditionally retrieve from User Stores
        dense_usr_docs = []
        bm25_usr_docs = [] 

        # Only proceed with user stores if at least user_id and kb_id are provided
        if user_id and kb_id and kb_document_id:
            # Build the Chroma-specific native filter dynamically
            filter_conditions = [
                {"user_id": user_id},
                {"kb_id": kb_id},
                {"kb_document_id": kb_document_id}
            ]

            # Chroma requires a specific format: direct dict for 1 condition, $and for multiple
            if len(filter_conditions) > 1:
                chroma_filter = {"$and": filter_conditions}
            else:
                chroma_filter = filter_conditions[0]

            # Fetch User Docs
            dense_usr_docs = self.dense_user.vectorstore.similarity_search(
                query, k=config.DENSE_K, filter=chroma_filter
            )
            
            # Pass document_id to BM25 only if it exists (assuming your BM25 implementation handles None gracefully)
            bm25_usr_docs = self.bm25_user.invoke(
                query, 
                user_id=user_id, 
                kb_id=kb_id, 
                document_id=kb_document_id
            )

        # Combine all retrieved documents
        all_docs = dense_int_docs + dense_usr_docs + bm25_int_docs + bm25_usr_docs

        seen_contents = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

        if not unique_docs:
            return []

        pairs = [(query, doc.page_content) for doc in unique_docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:config.RERANK_TOP_K]]

    
    def debug_sources(self, query: str, user_id: str , kb_id: str, kb_document_id: str) -> Tuple[List[Document], List[Document]]:
        """
        Return BM25-only docs and dense-only docs for a query, before ensemble fusion.
        Only for debugging / analysis.
        """
        # bm25 and dense are both retrievers
        dense_int_docs = self.dense_internal.invoke(query)
        dense_usr_docs = self.dense_user.invoke(query)

        bm25_int_docs = self.bm25_internal.invoke(query)
        bm25_usr_docs = self.bm25_user.invoke(query)
        return dense_int_docs, dense_usr_docs, bm25_int_docs, bm25_usr_docs

# ─────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────

class RAGPipeline:

    def __init__(self, collection_name: str = None, embedding_model: str = None, embeddings = None,):
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
        self.retriever: HybridRerankRetriever | None = None
        

    # ─────────────────────────────────────────
    # Internal: restore state from disk (server restart path)
    # ─────────────────────────────────────────

    def _load_docs_from_store(self) -> None:
        """
        Restore self.lc_docs from the persisted Chroma collection.
        Chroma writes to disk automatically (persist_directory), so all
        previously ingested documents survive server restarts.
        """
        result = self.vector_store._collection.get(
            include=["documents", "metadatas"]
        )
        self.lc_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
            if text
        ]

    
    # ─────────────────────────────────────────
    # Metadata Sanitization
    # ─────────────────────────────────────────

    def sanitize_metadata(self, meta: dict) -> dict:
        clean = {}
        for key, value in meta.items():
            if value is None:
                continue                          # skip None
            elif isinstance(value, list):
                if len(value) == 0:
                    continue                      # skip empty lists
                else:
                    clean[key] = ", ".join(str(v) for v in value)  # convert list → string
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value                # keep valid types
            else:
                clean[key] = str(value)           # coerce anything else to string
        return clean
    
    # ─────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────

    def process_json(self, json_path: str, store_docs: bool = False) -> None:
        # self.vector_store.reset_collection()
        self.lc_docs.clear()

        batch_docs = []
        batch_ids = []

        with open(json_path, "rb") as f:
            for entry in ijson.items(f, "item"):
                meta = entry.get("metadata", {})

                meta = self.sanitize_metadata(meta)

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

    
class LLMGenerator:

    def __init__(self):

        self.llm = ChatGroq(
            model= config.GROQ_MODEL,
            temperature= config.GROQ_TEMPERATURE,
            max_tokens= config.GROQ_MAX_TOKENS,
            streaming= False
        )

    def generate_answer(self, query: str, formatted_chunks: list, chat_history: list) -> str:
        # Combine the chunks into a single text block
        chunks_string = "\n\n".join(formatted_chunks)
        
        messages = [
            SystemMessage(content=system_template), # Your RAG system template
            HumanMessage(content=user_template.format(
                chunks=chunks_string,
                query=query,
                chat_history=chat_history
            ))
        ]
        response = self.llm.invoke(messages)
        return response.content

if __name__ == "__main__":
    # Quick local test
    # pipeline = RAGPipeline()
    bm25= BM25sDiskRetriever(index_path=str(config.BM25_INDEX_DIR / "0d3493be-c6b9-47dd-9387-7301b812b52a"))
    

    query = "What are the powers available to a court for enforcing execution of a decree under Section 51 of the Code of Civil Procedure?"
    answer = bm25.invoke(query,user_id= "80379425-7a6e-49b4-b8db-56341cb66c43", kb_id= "0d3493be-c6b9-47dd-9387-7301b812b52a") #document_id= "24cfd42d-dd3b-4ae3-a706-3d442addc5e7")

    print("\nANSWER:\n", answer)
    # print("\nSOURCES:\n", sources)