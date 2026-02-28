"""
Chat Store.

Responsibilities:
  - Append messages to chat/messages.jsonl (one JSON object per line)
  - Read full chat history for a notebook
  - Each message has: role, content, timestamp
  - Assistant messages also have: citations, rag_technique, timing metrics
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from storage.notebook_store import touch_notebook
from utils.config import DATA_ROOT
from utils.security import safe_path, sanitize_username

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chat_dir(username: str, notebook_id: str) -> Path:
    """Resolve …/users/<username>/notebooks/<notebook_id>/chat/"""
    base = DATA_ROOT / "users" / sanitize_username(username) / "notebooks"
    nb_dir = safe_path(base, notebook_id)
    chat_dir = nb_dir / "chat"
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir


def _messages_file(username: str, notebook_id: str) -> Path:
    return _chat_dir(username, notebook_id) / "messages.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_message(message: dict) -> dict:
    """
    Ensure the message dict has required fields and add defaults for optional ones.

    Required:
        role    – "user" | "assistant"
        content – non-empty string

    Auto-added if missing:
        timestamp – current UTC ISO 8601

    Returns a shallow copy so the caller's original dict is never mutated.

    Raises:
        ValueError – missing/invalid role or content
    """
    if "role" not in message:
        raise ValueError("message must include a 'role' field.")
    if message["role"] not in ("user", "assistant"):
        raise ValueError(
            f"role must be 'user' or 'assistant', got '{message['role']}'"
        )
    if "content" not in message or not str(message["content"]).strip():
        raise ValueError("message must include a non-empty 'content' field.")

    msg = dict(message)                      # shallow copy — never mutate caller's dict
    if "timestamp" not in msg:
        msg["timestamp"] = _now_iso()

    return msg


# ===========================================================================
# Public API  —  signatures match the stub exactly
# ===========================================================================

def append_message(username: str, notebook_id: str, message: dict) -> None:
    """
    Append a single message to the chat history.

    The message dict must contain at minimum::

        {
            "role":    "user" | "assistant",
            "content": str,
        }

    Optional fields (preserved as-is if present)::

        {
            "timestamp":     str,         # ISO 8601 UTC — added automatically if absent
            # assistant messages only:
            "citations":     list[dict],  # [{"source": str, "chunk": str, "score": float}]
            "rag_technique": str,         # e.g. "naive" | "rerank" | "hyde"
            "timing": {
                "retrieval_ms":  float,
                "generation_ms": float,
                "total_ms":      float,
            }
        }

    Args:
        username    – HuggingFace username
        notebook_id – UUID of the target notebook
        message     – dict conforming to the schema above

    Raises:
        ValueError – missing or invalid role / content
        OSError    – underlying filesystem failure
    """
    msg = _validate_message(message)
    path = _messages_file(username, notebook_id)

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
    try:
      touch_notebook(username, notebook_id)
    except KeyError:
      pass

def get_history(username: str, notebook_id: str) -> list[dict]:
    """
    Read and return the full chat history in chronological order.

    Returns an empty list if no history exists yet for this notebook.

    Each item matches the message schema described in :func:`append_message`.

    Args:
        username    – HuggingFace username
        notebook_id – UUID of the notebook

    Returns:
        List of message dicts, oldest first.
    """
    path = _messages_file(username, notebook_id)
    if not path.exists():
        return []

    messages: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupted lines rather than crashing the whole session.
                # A partial write during a crash can leave one bad line; all
                # preceding messages are still intact and readable.
                continue

    return messages


# ---------------------------------------------------------------------------
# Convenience helpers  (used by core/rag.py and the Gradio UI)
# Not in the stub, but needed by other modules.
# ---------------------------------------------------------------------------

def get_history_for_llm(username: str, notebook_id: str, window: int = 10) -> list[dict]:
    """
    Return the last *window* messages formatted for an LLM chat API call.

    Strips internal-only fields (citations, rag_technique, timing) so only
    ``role`` and ``content`` are sent to the model.

    Args:
        window – number of most-recent messages to include (default 10)

    Returns:
        ``[{"role": str, "content": str}, ...]``
    """
    history = get_history(username, notebook_id)
    recent = history[-window:] if window else history
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def clear_history(username: str, notebook_id: str) -> int:
    """
    Delete all messages for a notebook.

    Returns:
        Number of messages deleted (0 if no history existed).
    """
    path = _messages_file(username, notebook_id)
    if not path.exists():
        return 0
    count = len(get_history(username, notebook_id))
    path.unlink()
    return count