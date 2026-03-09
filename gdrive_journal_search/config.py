"""Central configuration for gdrive-journal-search."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
SYNC_STATE_FILE = BASE_DIR / "sync_state.json"
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

# Embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Reranking model (cross-encoder, runs locally)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Ollama LLM settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

# ChromaDB collection name
CHROMA_COLLECTION = "journal"

# Chunking settings
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks

# Retrieval settings
TOP_K_RETRIEVE = 75     # candidates from each method (vector + BM25 → ~150 merged)
TOP_K_FINAL = 20        # chunks kept after reranking and sent to LLM

# Sync settings
FETCH_WORKERS = 10      # parallel threads for Drive doc exports
