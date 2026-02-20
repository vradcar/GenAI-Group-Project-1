"""
Vector Store — ChromaDB wrapper.

Responsibilities:
  - Create / get / delete per-notebook ChromaDB collections
  - Add documents (chunks + embeddings + metadata) to a collection
  - Query a collection by embedding similarity (top-k)
  - Use persistent storage under <notebook>/chroma/
"""


def get_or_create_collection(username: str, notebook_id: str):
    """Get or create a ChromaDB collection for a notebook."""
    # TODO
    pass


def add_documents(username: str, notebook_id: str, chunks: list[str], metadatas: list[dict]) -> None:
    """Embed and add chunks to the notebook's collection."""
    # TODO
    pass


def query_collection(username: str, notebook_id: str, query_text: str, n_results: int = 5) -> dict:
    """Query the collection and return top-k results."""
    # TODO
    pass


def delete_collection(username: str, notebook_id: str) -> None:
    """Delete the entire collection for a notebook."""
    # TODO
    pass
