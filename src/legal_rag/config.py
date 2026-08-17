from pathlib import Path

BASE_DIR = Path(__file__).parent

INTERMEDIATE_DIR = BASE_DIR / "data/intermediate"
PROCESSED_DIR    = BASE_DIR / "data/processed"
RAW_DIR          = BASE_DIR / "data/raw"

# ── Parsing ───────────────────────────────────────────────────────────────────
CHUNK_JSON = str(INTERMEDIATE_DIR / "final_chunk.json")
KEY_DATA_JSON = str(PROCESSED_DIR / "extracted_data.json")
PAGES_PER_BATCH  = 8
BATCH_DIR        = INTERMEDIATE_DIR / "batches"
BATCH_DIR_2        = INTERMEDIATE_DIR / "batches_2"
BATCH_DIR_3        = INTERMEDIATE_DIR / "batches_3"
INDEX_BATCH_DIR    = INTERMEDIATE_DIR / "index_batch"

# ── RAG / Vector store ────────────────────────────────────────────────────────
VECTORSTORE_DIR  = BASE_DIR / "resource/vectorstore"
BM25_INDEX_DIR= BASE_DIR / "resource/bm25_index"
COLLECTION_NAME  = "rag_legaldoc"
EVAL_COLLECTION_NAME = "rag_legaldoc_eval"
# EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE       = 100
EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_MODEL       = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS  = 3000

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_LLM_MODEL       = "llama-3.3-70b-versatile"
GT_LLM_MODEL         = "llama-3.3-70b-versatile"
QUERY_BATCH_SIZE     = 8
QUERY_BATCH_OVERLAP  = 3
RERANK_TOP_K         = 3
BM25_K               = 10
DENSE_K              = 10

# ── LLM as Judge ────────────────────────────────────────────────────────────────
LLM_AS_JUDGE       = "openai/gpt-oss-120b"
JUDGE_TEMPERATURE = 0.1
PASS_THRESHOLD = 0.75
RETRY_THRESHOLD = 0.50
MIN_RELEVANT_CHUNKS_FOR_PASS = 1

# ── Routing ────────────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.8

# ── User Workspace ────────────────────────────────────────────────────────────────
USER_KB_DIR = BASE_DIR / "user_workspace/local_kb_storage"