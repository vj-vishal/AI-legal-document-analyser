from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple, Optional
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
import torch
import src.legal_rag.config as config
from langchain_classic.schema import SystemMessage, HumanMessage
# from langchain.callbacks.base import BaseCallbackHandler

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

### Behavior for Knowledge/QA Queries

If the request is a Knowledge/QA Query, do the following:

1. Answer only from `<context_chunks>`
- Provide a clear, direct, and objective legal answer based solely on the retrieved text.
- If multiple chunks are relevant, synthesize them carefully without adding anything not supported by the text.
- If the chunks contain conflicting information, state that clearly and cite each conflicting source.

2. Mandatory citation
- Every material legal statement, rule, proposition, conclusion, or factual claim must include an inline citation.
- The citation must appear at the end of the relevant sentence or paragraph.
- Use a consistent citation format and include all available metadata fields from the source chunk.
- If metadata fields vary by chunk, include every available key-value pair for that chunk without inventing missing metadata.

3. Citation format
- Use this format:
  `[Source: <all available metadata key-value pairs from the chunk>]`
- Example:
  `[Source: title=Indian Contract Act Summary; section=Offer and Acceptance; page=12; chunk_id=ch_004; jurisdiction=India]`

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
- Extract from the user’s request any values already provided.

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

1. Read the user query.
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
- Extract all information already supplied by the user.
- If required fields are missing, ask for all missing requirements in one consolidated response.
- If enough information is available, produce the completed draft using the retrieved template as the baseline.
- Use brackets for any still-missing fields only where necessary.

6. If Mixed Query:
- First answer the legal question portion with citations.
- Then handle the drafting portion using the drafting workflow above.

### Output Rules

1. For Knowledge/QA Queries
- Be clear, objective, and concise.
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
- `<user_query>`: the user’s request
- `<context_chunks>`: the retrieved legal text and metadata

### Final fallback sentence

If the chunks do not contain enough information to answer the legal question or produce the requested draft/template workflow, output exactly:
`The retrieved legal documents do not contain sufficient information to fulfill this request.`
"""

user_template = """
<context_chunks>
{chunks}
</context_chunks>

<query>
{query}
</query>
"""


# ─────────────────────────────────────────────
# Debugging Callback
# ─────────────────────────────────────────────

# class LLMInspectorCallback(BaseCallbackHandler):
#     def on_llm_start(self, serialized, prompts, **kwargs):
#         print("\n" + "="*60)
#         print(f"[LLM INPUT] Model: {serialized.get('name')}")
#         print(f"[DOCS + PROMPT SENT]:\n")
#         for i, p in enumerate(prompts):
#             print(f"--- Prompt {i+1} ---\n{p}\n")
#         print("="*60)

#     def on_llm_end(self, response, **kwargs):
#         print(f"\n[LLM OUTPUT]: {response.generations[0][0].text[:300]}...")


# ─────────────────────────────────────────────
# Per-User BM25 Retriever with Disk Persistence
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Hybrid + Rerank Retriever
# ─────────────────────────────────────────────

class HybridRerankRetriever(BaseRetriever):
    bm25: BM25sDiskRetriever
    dense: Any
    ensemble: EnsembleRetriever
    reranker: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_components(
        cls,
        bm25: BM25sDiskRetriever,   
        vector_store: Chroma
    ) -> "HybridRerankRetriever":
        """
        Build the hybrid retriever from a pre-built BM25sDiskRetriever
        and the user's Chroma vector store.
        Accepts either a freshly built or disk-loaded BM25 retriever.
        """
        dense = vector_store.as_retriever(
            search_kwargs={"k": config.DENSE_K}
        )
        ensemble = EnsembleRetriever(retrievers=[bm25, dense])
        # reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # reranker = FlagReranker(
        #     "BAAI/bge-reranker-v2-m3",
        #     use_fp16=True,       
        #     device="cuda"        
        # )

        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",                         # or "cpu"
            max_length=1024,                       # recommended by BAAI [web:170]
            default_activation_function=None,
            model_kwargs={"torch_dtype": torch.float16},  # fp16 to save VRAM
)

        return cls(
            bm25=bm25,
            dense=dense,
            ensemble=ensemble,
            reranker=reranker
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        docs = self.ensemble.invoke(query)

        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:config.RERANK_TOP_K]]

        # added just for checking the scores
        # top = ranked[:config.RERANK_TOP_K]

        # out_docs = []
        # for doc, score in top:
        #     meta = dict(doc.metadata) if doc.metadata else {}
        #     meta["rerank_score"] = float(score)
        #     doc.metadata = meta
        #     out_docs.append(doc)

        # return out_docs
    
    def debug_sources(self, query: str) -> Tuple[List[Document], List[Document]]:
        """
        Return BM25-only docs and dense-only docs for a query, before ensemble fusion.
        Only for debugging / analysis.
        """
        # bm25 and dense are both retrievers
        bm25_docs = self.bm25.invoke(query)
        dense_docs = self.dense.invoke(query)
        return bm25_docs, dense_docs

# ─────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────

class RAGPipeline:

    def __init__(self, collection_name: str = None, embedding_model: str = None, embeddings = None,):
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
        self.retriever: HybridRerankRetriever | None = None
        # self.chain: RetrievalQAWithSourcesChain | None = None

        # Auto-recovery on server restart:
        # If the user's BM25 index exists on disk, Chroma already has their
        # vectors too. Reload both so queries work immediately without re-uploading.
        # if (Path(self._bm25_path) / "corpus_docs.pkl").exists():
        #     self._load_docs_from_store()
        #     self._load_retriever_and_chain()

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

    
class LLMGenerator:

    def __init__(self):

        self.llm = ChatGroq(
            model= config.GROQ_MODEL,
            temperature= config.GROQ_TEMPERATURE,
            max_tokens= config.GROQ_MAX_TOKENS,
            streaming= False
        )

    def generate_answer(self, query: str, formatted_chunks: list) -> str:
        # Combine the chunks into a single text block
        chunks_string = "\n\n".join(formatted_chunks)
        
        messages = [
            SystemMessage(content=system_template), # Your RAG system template
            HumanMessage(content=user_template.format(
                chunks=chunks_string,
                query=query
            ))
        ]
        response = self.llm.invoke(messages)
        return response.content

if __name__ == "__main__":
    # Quick local test
    pipeline = RAGPipeline()
    

    query = "What are the specific financial and administrative responsibilities of the Secretary of the Committee under Regulation 8?"
    answer = pipeline.generate_answer(query)

    print("\nANSWER:\n", answer)
    # print("\nSOURCES:\n", sources)