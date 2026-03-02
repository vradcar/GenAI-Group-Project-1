---
title: StudyPod
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.8.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# StudyPod — NotebookLM Clone

> ITCS 5010 GenAI Group Project 1

A Gradio web app where users upload documents (PDF, PPTX, TXT, or URLs), chat with them via RAG, and generate artifacts — structured reports, quizzes, and AI-narrated podcasts.

## Team
- Varad Paradkar
- Nidhi Shah
- Brinda Chinnaraji
- Sai Kiran Jagini
- Shar Adhiambo

---

## Features

- **Multi-format ingestion** — Upload PDF, PPTX, TXT files or paste a URL; text is extracted, chunked, and embedded automatically.
- **RAG Chat** — Ask questions about your documents with four retrieval techniques:
  | Technique | Description |
  |-----------|-------------|
  | **Naive** | Cosine-similarity top-K retrieval |
  | **HyDE** | LLM generates a hypothetical answer, embeds it, then retrieves similar real chunks |
  | **Reranking** | Retrieves a broad candidate set, then LLM scores each for relevance |
  | **Multi-Query** | LLM generates multiple query variants, retrieves for each, fuses results via Reciprocal Rank Fusion |
- **Artifact generation**
  - 📄 **Report** — Structured Markdown summary with executive summary, key concepts, and conclusions
  - ❓ **Quiz** — Mix of MCQ, short-answer, and True/False questions with an answer key
  - 🎙️ **Podcast** — Conversational transcript between two hosts + TTS audio via edge-tts
- **Per-notebook isolation** — Each notebook gets its own ChromaDB collection, chat history (JSONL), and artifact storage.
- **OAuth login** — Hugging Face OAuth via Gradio; all data scoped per user.
- **LLM fallback** — Primary model (Llama 3.3 70B) with automatic retry + fallback to 8B on failure.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/studypod.git
cd studypod
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

Copy the example and add your Groq API key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your-key-here
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key ([console.groq.com](https://console.groq.com)) |
| `DATA_DIR` | No | `data/` | Root directory for all user data |

### 3. Run

```bash
python app.py
```

The app launches at `http://localhost:7860` (or the port Gradio selects).

---

## Development

```bash
# Lint
ruff check .

# Run all tests (236 tests)
pytest tests/ -v

# Run a single test file
pytest tests/test_security.py -v
```

---

## Project Structure

```
app.py                  # Entry point — Gradio UI + callback wiring
core/
  ingestion.py          # Text extraction, chunking, embedding
  rag.py                # Query → retrieve → prompt → respond (4 techniques)
  artifacts.py          # Report / quiz / podcast generation
  llm_client.py         # Groq API wrapper with retry + fallback
  models.py             # Shared data classes (LLMResponse, Citation, etc.)
storage/
  notebook_store.py     # Notebook CRUD (index.json, per-notebook dirs)
  chat_store.py         # Chat history (JSONL append/read)
  vector_store.py       # ChromaDB collection management
  artifact_store.py     # Save / list / retrieve generated artifacts
utils/
  config.py             # Env vars, model names, constants
  security.py           # Path validation, input sanitization
  extractors.py         # File-type-specific text extraction (PDF, PPTX, TXT, URL)
tests/                  # 236 unit + integration tests
docs/
  proposal.tex          # Initial project proposal (LaTeX)
```

### Per-notebook data layout

```
data/users/<username>/notebooks/<uuid>/
  metadata.json
  files_raw/          # Original uploaded files
  files_extracted/    # Extracted plain text
  chroma/             # ChromaDB persistent storage
  chat/               # messages.jsonl
  artifacts/
    reports/          # report_<timestamp>.md
    quizzes/          # quiz_<timestamp>.md
    podcasts/         # transcript_<timestamp>.md + podcast_<timestamp>.mp3
```

---

## Tech Stack

| Layer | Tool | Details |
|-------|------|---------|
| Frontend | Gradio 6.x | OAuth login, chat, file upload, audio player |
| LLM | Groq API | Llama 3.3 70B primary, 8B fallback (OpenAI-compatible SDK) |
| Embeddings | sentence-transformers | `all-MiniLM-L6-v2` — 384-dim, runs locally on CPU |
| Vector DB | ChromaDB | Persistent, per-notebook collections with cosine similarity |
| TTS | edge-tts | `en-US-AriaNeural` voice for podcast audio |
| Hosting | Hugging Face Spaces | Auto-deployed via GitHub Actions |
| CI/CD | GitHub Actions | Lint (`ruff`) → Test (`pytest`) → Deploy to HF Space |

---

## Deployment

The GitHub Actions workflow (`.github/workflows/deploy.yml`) runs on push to `main`:

1. **Lint** — `ruff check .`
2. **Test** — `pytest tests/ -v`
3. **Deploy** — Push to Hugging Face Space

### Required GitHub settings

| Setting | Type | Value |
|---------|------|-------|
| `HF_TOKEN` | Repository secret | Hugging Face write token |
| `HF_SPACE_NAME` | Repository variable | e.g. `your-org/studypod` |

### Required HF Space settings

| Setting | Type | Value |
|---------|------|-------|
| `GROQ_API_KEY` | Space secret | Your Groq API key |
