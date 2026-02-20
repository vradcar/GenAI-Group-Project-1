"""
Ingestion Pipeline.

Responsibilities:
  - Accept uploaded files (PDF, PPTX, TXT) or URLs
  - Validate file type (whitelist) and size (50 MB max)
  - Save raw file to files_raw/
  - Extract text via extractors.py and save to files_extracted/
  - Chunk text with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  - Embed chunks locally via all-MiniLM-L6-v2
  - Upsert embeddings + metadata into ChromaDB
  - Update notebook metadata (source count, timestamp)
"""


def ingest_file(username: str, notebook_id: str, file_path: str) -> dict:
    """Ingest a single uploaded file into the notebook's vector store."""
    # TODO
    pass


def ingest_url(username: str, notebook_id: str, url: str) -> dict:
    """Fetch URL content, extract text, chunk, embed, and store."""
    # TODO
    pass
