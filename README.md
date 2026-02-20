# StudyPod — NotebookLM Clone

> ITCS 5010 GenAI Group Project 1

## Team
- Varad Paradkar
- Nidhi Shah
- Brinda Chinnaraji
- Sai Kiran Jagini
- Shar Adhiambo

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Project Structure

```
app.py                  # Entry point — Gradio UI + callback wiring
core/
  ingestion.py          # Text extraction, chunking, embedding
  rag.py                # Query → retrieve → prompt → respond
  artifacts.py          # Report / quiz / podcast generation
  llm_client.py         # Groq API wrapper with retry logic
storage/
  notebook_store.py     # Notebook CRUD (index.json, directories)
  chat_store.py         # Chat history (JSONL append/read)
  vector_store.py       # ChromaDB collection management
  artifact_store.py     # Save / list / retrieve artifacts
utils/
  config.py             # Env vars, model names, constants
  security.py           # Path validation, input sanitization
  extractors.py         # File-type-specific text extraction
tests/                  # Unit tests
docs/
  proposal.tex          # Initial plan (LaTeX)
```

## Tech Stack

| Layer      | Tool                        |
|------------|-----------------------------|
| Frontend   | Gradio                      |
| LLM        | Groq API (Llama 3.1 70B)    |
| Embeddings | sentence-transformers (local)|
| Vector DB  | ChromaDB                    |
| TTS        | edge-tts                    |
| Hosting    | Hugging Face Spaces         |
| CI/CD      | GitHub Actions              |
