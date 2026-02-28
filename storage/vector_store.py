"""
Vector Store — ChromaDB wrapper.

Responsibilities:
  - Create / get / delete per-notebook ChromaDB collections
  - Add documents (chunks + embeddings + metadata) to a collection
  - Query a collection by embedding similarity (top-k)
  - Use persistent storage under <notebook>/chroma/
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from storage.notebook_store import get_chroma_dir
from utils.config import EMBEDDING_MODEL, EMBEDDING_DEVICE

# ---------------------------------------------------------------------------
# Embedding function (shared across all collections)
#
# sentence-transformers runs locally — no API key needed.
# We build it once and reuse it so the model isn't reloaded on every call.
# ---------------------------------------------------------------------------

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    device=EMBEDDING_DEVICE,
)

# ---------------------------------------------------------------------------
# ChromaDB client cache
#
# One PersistentClient per chroma/ directory.
# Reusing the same client object avoids SQLite "database is locked" errors
# when multiple Gradio callbacks hit the same notebook concurrently.
# ---------------------------------------------------------------------------

_clients: dict[str, chromadb.PersistentClient] = {}


def _get_client(chroma_dir: Path) -> chromadb.PersistentClient:
    key = str(chroma_dir.resolve())
    if key not in _clients:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        _clients[key] = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
    return _clients[key]


def _collection_name(notebook_id: str) -> str:
    """
    Derive a ChromaDB-safe collection name from a notebook UUID.

    ChromaDB rules: 3-63 chars, alphanumeric start/end, only [a-z0-9_-].
    UUIDs contain hyphens and are 36 chars — strip hyphens and prefix 'nb_'.
    """
    return ("nb_" + notebook_id.replace("-", ""))[:63]


def _chunk_id(source: str, index: int) -> str:
    """
    Stable, collision-resistant chunk ID derived from (source, index).

    Using a deterministic ID means calling add_documents twice with the same
    file is idempotent — ChromaDB upsert replaces rather than duplicates.
    """
    raw = f"{source}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


# ===========================================================================
# Public API  —  signatures match the stub exactly
# ===========================================================================

def get_or_create_collection(username: str, notebook_id: str):
    """
    Get or create a ChromaDB collection for a notebook.

    Each notebook gets its own isolated collection backed by a persistent
    SQLite + HNSW index under ``<notebook>/chroma/``.

    The collection uses cosine similarity (better than L2 for text embeddings)
    and the shared sentence-transformers embedding function.

    Args:
        username    – HuggingFace username
        notebook_id – UUID of the notebook

    Returns:
        ``chromadb.Collection`` ready for upsert / query.
    """
    chroma_dir = get_chroma_dir(username, notebook_id)
    client = _get_client(chroma_dir)

    return client.get_or_create_collection(
        name=_collection_name(notebook_id),
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    username: str,
    notebook_id: str,
    chunks: list[str],
    metadatas: list[dict],
) -> None:
    """
    Embed and add chunks to the notebook's collection.

    Uses ``upsert`` so calling this twice with the same source file is safe —
    existing chunks are updated rather than duplicated.  Chunk IDs are derived
    deterministically from ``metadata["source"]`` + chunk index.

    Args:
        username    – HuggingFace username
        notebook_id – UUID of the notebook
        chunks      – list of text strings to embed and store
        metadatas   – parallel list of metadata dicts; each should contain
                      at minimum ``{"source": "<filename or url>"}``

    Raises:
        ValueError – chunks and metadatas have different lengths, or either is empty
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")
    if len(chunks) != len(metadatas):
        raise ValueError(
            f"chunks ({len(chunks)}) and metadatas ({len(metadatas)}) must have the same length."
        )

    collection = get_or_create_collection(username, notebook_id)

    ids = [
        _chunk_id(metadatas[i].get("source", str(i)), i)
        for i in range(len(chunks))
    ]

    collection.upsert(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )


def query_collection(
    username: str,
    notebook_id: str,
    query_text: str,
    n_results: int = 5,
) -> dict:
    """
    Query the collection and return top-k results.

    Embeds *query_text* using the same sentence-transformers model used at
    index time, then performs an approximate nearest-neighbour search.

    Args:
        username    – HuggingFace username
        notebook_id – UUID of the notebook
        query_text  – natural-language query string
        n_results   – number of results to return (capped at collection size)

    Returns:
        Raw ChromaDB result dict::

            {
                "ids":        [[str, ...]],       # chunk IDs
                "documents":  [[str, ...]],       # chunk texts
                "metadatas":  [[dict, ...]],      # per-chunk metadata
                "distances":  [[float, ...]],     # cosine distances (lower = more similar)
                "embeddings": None,               # not returned by default
            }

        Returns a dict with empty inner lists if the collection is empty.

    Raises:
        ValueError – query_text is blank
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be blank.")

    collection = get_or_create_collection(username, notebook_id)
    count = collection.count()

    if count == 0:
        # Return the same shape ChromaDB would return — callers can check
        # results["ids"][0] == [] without special-casing an empty collection.
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    actual_n = min(n_results, count)

    return collection.query(
        query_texts=[query_text],
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )


def delete_collection(username: str, notebook_id: str) -> None:
    """
    Delete the entire collection for a notebook.

    Called when a notebook is deleted.  Safe to call if the collection does
    not exist yet — the error is swallowed silently.

    Args:
        username    – HuggingFace username
        notebook_id – UUID of the notebook
    """
    chroma_dir = get_chroma_dir(username, notebook_id)
    if not chroma_dir.exists():
        return

    client = _get_client(chroma_dir)
    name = _collection_name(notebook_id)

    try:
        client.delete_collection(name)
    except Exception:
        pass  # already absent — not an error


# ---------------------------------------------------------------------------
# Convenience helpers  (used by core/ingestion.py and core/rag.py)
# Not in the stub, but needed by other modules.
# ---------------------------------------------------------------------------

def collection_count(username: str, notebook_id: str) -> int:
    """Return the total number of chunks indexed in this notebook."""
    return get_or_create_collection(username, notebook_id).count()


def list_sources(username: str, notebook_id: str) -> list[str]:
    """
    Return the unique source names currently indexed in this notebook.

    Useful for the Gradio UI to show which files have been ingested.
    """
    collection = get_or_create_collection(username, notebook_id)
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    seen: set[str] = set()
    for m in all_meta:
        src = m.get("source")
        if src:
            seen.add(src)
    return sorted(seen)


def delete_source(username: str, notebook_id: str, source: str) -> int:
    """
    Remove all chunks whose ``metadata["source"]`` equals *source*.

    Called when a user removes a single file from a notebook without
    deleting the whole notebook.

    Returns:
        Number of chunks deleted.
    """
    collection = get_or_create_collection(username, notebook_id)
    hits = collection.get(where={"source": source})
    ids = hits["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)