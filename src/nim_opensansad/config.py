import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}. Copy .env.example → .env and fill it in.")
    return val


# NVIDIA NIM (rerank + LLM only — embeddings are local)
NVIDIA_API_KEY = _require("NVIDIA_API_KEY")
LLM_MODEL      = os.getenv("LLM_MODEL",     "meta/llama-3.1-70b-instruct")
RERANK_MODEL   = os.getenv("RERANK_MODEL",  "nvidia/llama-3.2-nv-rerankqa-1b-v2")

# Local embedding model (runs on CPU, no API calls)
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-large-v2")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))

# Milvus
MILVUS_URI      = os.getenv("MILVUS_URI",      "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "opensansad")

# Embedding
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Retrieval
TOP_K        = int(os.getenv("TOP_K",        "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
