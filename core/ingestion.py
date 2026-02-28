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

import shutil
from pathlib import Path
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from storage import vector_store
from utils import extractors
from utils.config import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_FILE_SIZE_MB,
    USERS_DIR,
)
from utils.security import sanitize_filename, validate_path


def _notebook_dir(username: str, notebook_id: str) -> Path:
    """Return the base directory for a notebook."""
    return Path(USERS_DIR) / username / notebook_id


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def ingest_file(username: str, notebook_id: str, file_path: str) -> dict:
    """
    Ingest a single uploaded file into the notebook's vector store.

    Steps:
      1. Validate extension against the allowed whitelist.
      2. Validate file size against the 50 MB limit.
      3. Sanitize the filename.
      4. Copy the raw file into files_raw/.
      5. Extract text with the appropriate extractor.
      6. Save extracted text to files_extracted/.
      7. Chunk the text.
      8. Upsert chunks + metadata into ChromaDB via vector_store.

    Returns a status dict: {"status", "source", "chunks", "extracted_chars"}.
    Raises ValueError for invalid extension, oversized file, or empty extraction.
    """
    src = Path(file_path)
    ext = src.suffix.lower()

    # Validate file extension
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Validate file size
    size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File size ({size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )

    # Sanitize filename
    safe_name = sanitize_filename(src.name)

    # Set up directory structure and copy raw file
    nb_dir = _notebook_dir(username, notebook_id)
    raw_dir = nb_dir / "files_raw"
    extracted_dir = nb_dir / "files_extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    raw_path = validate_path(str(raw_dir / safe_name), str(nb_dir))
    shutil.copy2(src, raw_path)

    # Extract text with the appropriate extractor
    if ext == ".pdf":
        text = extractors.extract_pdf(str(raw_path))
    elif ext == ".pptx":
        text = extractors.extract_pptx(str(raw_path))
    else:  # .txt
        text = extractors.extract_txt(str(raw_path))

    if not text.strip():
        raise ValueError("No text could be extracted from the file.")

    # Save extracted text
    extracted_path = validate_path(
        str(extracted_dir / (src.stem + ".txt")), str(nb_dir)
    )
    Path(extracted_path).write_text(text, encoding="utf-8")

    # Chunk text
    chunks = _chunk_text(text)

    # Build per-chunk metadata and upsert into ChromaDB
    metadatas = [
        {"source": safe_name, "chunk_index": i, "source_type": "file"}
        for i in range(len(chunks))
    ]
    vector_store.add_documents(username, notebook_id, chunks, metadatas)

    return {
        "status": "ok",
        "source": safe_name,
        "chunks": len(chunks),
        "extracted_chars": len(text),
    }


def ingest_url(username: str, notebook_id: str, url: str) -> dict:
    """
    Fetch URL content, extract text, chunk, embed, and store.

    Steps:
      1. Extract text from the URL via trafilatura.
      2. Derive a safe filename stem from the URL path.
      3. Save extracted text to files_extracted/.
      4. Chunk the text.
      5. Upsert chunks + metadata into ChromaDB via vector_store.

    Returns a status dict: {"status", "source", "chunks", "extracted_chars"}.
    Raises ValueError if the URL cannot be fetched or yields no text.
    """
   
    text = extractors.extract_url(url)

    if not text.strip():
        raise ValueError("No text could be extracted from the URL.")

    # Derive a safe filename stem from the URL
    parsed = urlparse(url)
    raw_stem = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc.replace(".", "_")
    safe_stem = sanitize_filename(raw_stem)[:80] or "webpage"

    # Set up directory and save extracted text
    nb_dir = _notebook_dir(username, notebook_id)
    extracted_dir = nb_dir / "files_extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = validate_path(
        str(extracted_dir / (safe_stem + "_url.txt")), str(nb_dir)
    )
    Path(extracted_path).write_text(text, encoding="utf-8")

    chunks = _chunk_text(text)

    metadatas = [
        {"source": url, "chunk_index": i, "source_type": "url"}
        for i in range(len(chunks))
    ]
    vector_store.add_documents(username, notebook_id, chunks, metadatas)

    return {
        "status": "ok",
        "source": url,
        "chunks": len(chunks),
        "extracted_chars": len(text),
    }
