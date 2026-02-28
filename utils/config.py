"""
Configuration — environment variables, model names, constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Groq / LLM ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_FALLBACK_MODEL = "llama-3.1-8b-instant"
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── RAG ──────────────────────────────────────────────────────
TOP_K = 5
RERANK_CANDIDATES = 20
MULTI_QUERY_VARIANTS = 3

# ── TTS ──────────────────────────────────────────────────────
TTS_VOICE = "en-US-AriaNeural"

# ── Storage ──────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "data")
DATA_ROOT = Path(DATA_DIR)
USERS_DIR = os.path.join(DATA_DIR, "users")

# ── Upload limits ────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".txt"}
