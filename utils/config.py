"""
Configuration — environment variables, model names, constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq / LLM ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.1-70b-versatile"
LLM_FALLBACK_MODEL = "llama-3.1-8b-instant"

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── RAG ──────────────────────────────────────────────────────
TOP_K = 5

# ── TTS ──────────────────────────────────────────────────────
TTS_VOICE = "en-US-AriaNeural"

# ── Storage ──────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "data")
USERS_DIR = os.path.join(DATA_DIR, "users")

# ── Upload limits ────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".txt"}
