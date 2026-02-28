"""
Vector Store — ChromaDB wrapper.

Responsibilities:
  - Create / get / delete per-notebook ChromaDB collections
  - Add documents (chunks + embeddings + metadata) to a collection
  - Query a collection by embedding similarity (top-k)
  - Use persistent storage under <notebook>/chroma/
"""

import os
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from utils.config import DATA_DIR, EMBEDDING_MODEL, TOP_K

logger = logging.getLogger(__name__)

# ── Embedding model (loaded once at module level) ────────────
try:
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model '{EMBEDDING_MODEL}'. "
        f"Ensure sentence-transformers is installed and the model is available. "
        f"Original error: {e}"
    ) from e


def _get_chroma_path(username: str, notebook_id: str) -> str:
    """Return the ChromaDB persistent storage path for a notebook."""
    return os.path.join(DATA_DIR, "users", username, "notebooks", notebook_id, "chroma")


def get_or_create_collection(username: str, notebook_id: str):
    """Get or create a ChromaDB collection for a notebook."""
    chroma_path = _get_chroma_path(username, notebook_id)
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(name="documents")


def add_documents(username: str, notebook_id: str, chunks: list[str], metadatas: list[dict]) -> None:
    """Embed and add chunks to the notebook's collection via idempotent upsert."""
    if not chunks:
        return

    collection = get_or_create_collection(username, notebook_id)
    embeddings = _embedding_model.encode(chunks).tolist()

    ids = []
    for meta in metadatas:
        source = meta.get("source_name", "unknown")
        idx = meta.get("chunk_index", 0)
        ids.append(f"{source}_{idx}")

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks for %s/%s", len(chunks), username, notebook_id)


def query_collection(username: str, notebook_id: str, query_text: str, n_results: int = TOP_K) -> dict:
    """Query the collection and return top-k results."""
    try:
        collection = get_or_create_collection(username, notebook_id)
    except Exception as e:
        logger.error("Failed to access vector store for %s/%s: %s", username, notebook_id, e)
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    query_embedding = _embedding_model.encode([query_text]).tolist()

    actual_n = min(n_results, collection.count())
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=actual_n,
    )
    return results


def delete_collection(username: str, notebook_id: str) -> None:
    """Delete the entire collection for a notebook."""
    chroma_path = _get_chroma_path(username, notebook_id)
    if not os.path.exists(chroma_path):
        return

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        client.delete_collection(name="documents")
    except Exception as e:
        logger.warning("Failed to delete collection for %s/%s: %s", username, notebook_id, e)
