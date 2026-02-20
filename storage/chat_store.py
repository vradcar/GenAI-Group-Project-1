"""
Chat Store.

Responsibilities:
  - Append messages to chat/messages.jsonl (one JSON object per line)
  - Read full chat history for a notebook
  - Each message has: role, content, timestamp
  - Assistant messages also have: citations, rag_technique, timing metrics
"""


def append_message(username: str, notebook_id: str, message: dict) -> None:
    """Append a single message to the chat history."""
    # TODO
    pass


def get_history(username: str, notebook_id: str) -> list[dict]:
    """Read and return the full chat history."""
    # TODO
    pass
