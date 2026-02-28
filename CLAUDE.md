# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StudyPod is a NotebookLM clone — a Gradio web app where users upload documents (PDF, PPTX, TXT, URLs), chat with them via RAG, and generate artifacts (reports, quizzes, podcasts). Built for ITCS 5010 GenAI Group Project.

## Commands

```bash
# Install dependencies (use Python 3.11+, venv at .venv/)
pip install -r requirements.txt

# Run the app
python app.py

# Lint
ruff check .

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_security.py -v

# Run a single test function
pytest tests/test_security.py::test_function_name -v
```

## Architecture

**Data flow:** Upload → `core/ingestion.py` (extract → chunk → embed) → ChromaDB → `core/rag.py` (query → retrieve → prompt → stream) → Gradio UI

**Three layers:**
- **`core/`** — Business logic. `ingestion.py` handles file processing pipeline; `rag.py` implements query with 4 techniques (naive, HyDE, reranking, multi-query); `artifacts.py` generates reports/quizzes/podcasts; `llm_client.py` wraps Groq API with retry + fallback from 70B to 8B model.
- **`storage/`** — Persistence. Each module handles one concern: `notebook_store.py` (CRUD + index.json), `chat_store.py` (JSONL append log), `vector_store.py` (ChromaDB collections), `artifact_store.py` (file-based). All scoped by `(username, notebook_id)`.
- **`utils/`** — Shared helpers. `config.py` loads env vars and defines all constants (model names, chunk sizes, limits). `security.py` handles path validation and input sanitization. `extractors.py` has per-filetype text extraction.

**Per-notebook directory layout** (under `data/users/<username>/<notebook-uuid>/`):
```
metadata.json, files_raw/, files_extracted/, chroma/, chat/, artifacts/
```

## Key Technical Details

- **LLM:** Groq API (OpenAI-compatible SDK) with Llama 3.1 70B primary, 8B fallback
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (local, not API)
- **Vector DB:** ChromaDB with persistent per-notebook storage
- **TTS:** edge-tts with `en-US-AriaNeural` voice for podcast generation
- **Auth:** Gradio OAuth (`gr.OAuthProfile`)
- **Storage writes:** Atomic (write-to-temp-then-rename) for JSON files
- **CI:** GitHub Actions runs `ruff check .` then `pytest tests/ -v` on push to main, deploys to Hugging Face Spaces

## Environment

Requires `GROQ_API_KEY` in `.env` (see `.env.example`). Data directory defaults to `data/` (override with `DATA_DIR` env var).

## Active Technologies
- Python 3.11+ + groq (OpenAI-compatible SDK), chromadb, (001-rag-llm-core)
- ChromaDB (persistent, per-notebook) + JSONL (chat history) (001-rag-llm-core)

## Recent Changes
- 001-rag-llm-core: Added Python 3.11+ + groq (OpenAI-compatible SDK), chromadb,
