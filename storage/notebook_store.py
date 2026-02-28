"""
Notebook Store.

Responsibilities:
  - Create / list / get / delete notebooks for a user
  - Manage index.json (notebook registry)
  - Create the directory structure for each notebook:
      <notebook-uuid>/
        metadata.json, files_raw/, files_extracted/,
        chroma/, chat/, artifacts/
  - Update metadata on changes
  - Use atomic writes (write-to-temp-then-rename) for JSON files
"""

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.config import DATA_ROOT
from utils.security import safe_path, sanitize_notebook_name, sanitize_username

#----------- logs-------
import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory layout constants
# ---------------------------------------------------------------------------

# Sub-directories created inside every new notebook
_NOTEBOOK_SUBDIRS: list[str] = [
    "files_raw",
    "files_extracted",
    "chroma",
    "chat",
    "artifacts/reports",
    "artifacts/quizzes",
    "artifacts/podcasts",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _user_notebooks_dir(username: str) -> Path:
    """…/data/users/<username>/notebooks/"""
    return DATA_ROOT / "users" / sanitize_username(username) / "notebooks"


def _index_path(username: str) -> Path:
    """…/notebooks/index.json  — registry of all notebook IDs + names."""
    return _user_notebooks_dir(username) / "index.json"


def _notebook_dir(username: str, notebook_id: str) -> Path:
    """…/notebooks/<notebook-uuid>/  — traversal-safe."""
    return safe_path(_user_notebooks_dir(username), notebook_id)


def _metadata_path(username: str, notebook_id: str) -> Path:
    """…/notebooks/<notebook-uuid>/metadata.json"""
    return _notebook_dir(username, notebook_id) / "metadata.json"


# ---------------------------------------------------------------------------
# Atomic JSON I/O
#
# Why atomic?  If the process is killed mid-write the old file is still
# intact.  We write to a sibling temp file, then os.replace() which is an
# atomic rename on POSIX (and best-effort on Windows).
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data) -> None:
    """
    Serialise *data* to JSON and write it to *path* atomically.

    Steps:
        1. Write to a temporary file in the same directory as *path*.
        2. os.replace(tmp, path)  — atomic on POSIX, best-effort on Windows.

    The temp file is always cleaned up even if an exception occurs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_",
        suffix=".json",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)          # atomic rename
    except Exception:
        # Clean up the orphaned temp file before re-raising
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json(path: Path, default):
    """
    Read and parse a JSON file.  Returns *default* if the file is absent
    or contains invalid JSON (graceful degradation, not silent corruption).
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# ---------------------------------------------------------------------------
# index.json helpers
#
# index.json is a list of lightweight summary records:
#   [{"id": str, "name": str, "created_at": str, "updated_at": str}, ...]
#
# Full metadata lives in each notebook's own metadata.json.
# ---------------------------------------------------------------------------

def _load_index(username: str) -> list[dict]:
    return _read_json(_index_path(username), default=[])


def _save_index(username: str, index: list[dict]) -> None:
    _atomic_write_json(_index_path(username), index)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Public API  —  signatures match the stub exactly
# ===========================================================================

def create_notebook(username: str, name: str) -> dict:
    """
    Create a new notebook and return its metadata.

    Metadata schema::

        {
            "id":         str,   # UUID-4
            "name":       str,   # sanitized display name
            "created_at": str,   # ISO 8601 UTC
            "updated_at": str,
        }

    Two files are written atomically:
      - ``index.json``   — updated registry entry
      - ``metadata.json``— full record inside the notebook directory

    Raises:
        ValueError   – blank or unsafe name
        RuntimeError – a notebook with the same name already exists
    """
    name = sanitize_notebook_name(name)

    index = _load_index(username)
    if any(nb["name"].lower() == name.lower() for nb in index):
        raise RuntimeError(f"A notebook named '{name}' already exists.")

    notebook_id = str(uuid.uuid4())
    now = _now()
    metadata = {
        "id":         notebook_id,
        "name":       name,
        "created_at": now,
        "updated_at": now,
    }

    # 1. Create sub-directory tree
    nb_dir = _notebook_dir(username, notebook_id)
    for sub in _NOTEBOOK_SUBDIRS:
        (nb_dir / sub).mkdir(parents=True, exist_ok=True)

    # 2. Write metadata.json inside the notebook directory (atomic)
    _atomic_write_json(_metadata_path(username, notebook_id), metadata)

    # 3. Append to index.json (atomic)
    index.append(metadata)
    _save_index(username, index)

    #logger info 
    logger.info(
    "Notebook created | user=%s notebook_id=%s name=%s",
    username,
    notebook_id,
    name,
)
    return metadata


def list_notebooks(username: str) -> list[dict]:
    """
    Return all notebooks for a user, sorted newest-first by *updated_at*.

    Reads from index.json — O(1) regardless of how many files each notebook
    contains.  Returns an empty list if the user has no notebooks yet.
    """
    index = _load_index(username)
    return sorted(
        index,
        key=lambda nb: nb.get("updated_at", nb.get("created_at", "")),
        reverse=True,
    )


def get_notebook(username: str, notebook_id: str) -> dict:
    """
    Return metadata for a single notebook.

    Reads from the notebook's own ``metadata.json`` (authoritative copy)
    rather than the index, so callers always get the freshest record.

    Raises:
        KeyError – notebook not found
    """
    path = _metadata_path(username, notebook_id)
    metadata = _read_json(path, default=None)

    if metadata is None:
        raise KeyError(
            f"Notebook '{notebook_id}' not found for user '{username}'."
        )

    return metadata


def delete_notebook(username: str, notebook_id: str) -> None:
    """
    Delete a notebook and all its data.

    Steps (in safe order):
        1. Verify the notebook exists.
        2. Remove the notebook directory tree from disk.
        3. Remove the notebook's entry from index.json (atomic write).

    Raises:
        KeyError – notebook not found
    """
    # 1. Verify existence before touching anything
    index = _load_index(username)
    updated_index = [nb for nb in index if nb["id"] != notebook_id]

    if len(updated_index) == len(index):
        raise KeyError(
            f"Notebook '{notebook_id}' not found for user '{username}'."
        )

    # 2. Delete directory tree (all files_raw, chroma, chat, artifacts, etc.)
    nb_dir = _notebook_dir(username, notebook_id)
    if nb_dir.exists():
        shutil.rmtree(nb_dir)

    # 3. Atomically update index.json
    _save_index(username, updated_index)
    
    #logger info 
    logger.info(
    "Notebook deleted | user=%s notebook_id=%s",
    username,
    notebook_id,
)

# ---------------------------------------------------------------------------
# Helpers used by other storage modules
# Not in the stub, but needed so chat_store / artifact_store don't
# re-implement path logic.
# ---------------------------------------------------------------------------

def update_notebook_name(username: str, notebook_id: str, new_name: str) -> dict:
    """
    Rename a notebook.  Updates both metadata.json and index.json atomically.

    Raises:
        KeyError     – notebook not found
        RuntimeError – new name already taken by another notebook
        ValueError   – blank or unsafe name
    """
    new_name = sanitize_notebook_name(new_name)
    index = _load_index(username)

    target = next((nb for nb in index if nb["id"] == notebook_id), None)
    if target is None:
        raise KeyError(f"Notebook '{notebook_id}' not found for user '{username}'.")

    if any(
        nb["name"].lower() == new_name.lower() and nb["id"] != notebook_id
        for nb in index
    ):
        raise RuntimeError(f"A notebook named '{new_name}' already exists.")

    now = _now()
    target["name"] = new_name
    target["updated_at"] = now

    # Keep metadata.json in sync
    metadata = _read_json(_metadata_path(username, notebook_id), default=dict(target))
    metadata["name"] = new_name
    metadata["updated_at"] = now

    _atomic_write_json(_metadata_path(username, notebook_id), metadata)
    _save_index(username, index)

    #logger info
    logger.info(
    "Notebook renamed | user=%s notebook_id=%s new_name=%s",
    username,
    notebook_id,
    new_name,
)
    return metadata


def touch_notebook(username: str, notebook_id: str) -> None:
    """
    Update *updated_at* to now in both metadata.json and index.json.

    Called by chat_store and artifact_store after every write so the
    'sorted by recently active' order in list_notebooks() stays accurate.

    Raises:
        KeyError – notebook not found
    """
    index = _load_index(username)
    target = next((nb for nb in index if nb["id"] == notebook_id), None)
    if target is None:
        raise KeyError(f"Notebook '{notebook_id}' not found for user '{username}'.")

    now = _now()
    target["updated_at"] = now

    metadata = _read_json(_metadata_path(username, notebook_id), default=dict(target))
    metadata["updated_at"] = now

    _atomic_write_json(_metadata_path(username, notebook_id), metadata)
    _save_index(username, index)
     
    #logger info 
    logger.debug(
    "Notebook touched | user=%s notebook_id=%s",
    username,
    notebook_id,
)

def get_notebook_dir(username: str, notebook_id: str) -> Path:
    """Return the root Path for a notebook directory."""
    return _notebook_dir(username, notebook_id)


def get_raw_dir(username: str, notebook_id: str) -> Path:
    return _notebook_dir(username, notebook_id) / "files_raw"


def get_extracted_dir(username: str, notebook_id: str) -> Path:
    return _notebook_dir(username, notebook_id) / "files_extracted"


def get_chroma_dir(username: str, notebook_id: str) -> Path:
    return _notebook_dir(username, notebook_id) / "chroma"


def get_chat_dir(username: str, notebook_id: str) -> Path:
    return _notebook_dir(username, notebook_id) / "chat"


def get_artifact_dir(username: str, notebook_id: str, artifact_type: str) -> Path:
    allowed = {"reports", "quizzes", "podcasts"}
    if artifact_type not in allowed:
        raise ValueError(f"artifact_type must be one of {allowed}, got '{artifact_type}'")
    return _notebook_dir(username, notebook_id) / "artifacts" / artifact_type

