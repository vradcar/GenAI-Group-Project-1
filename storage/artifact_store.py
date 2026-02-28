"""
Artifact Store.
Responsibilities:
  - Save generated artifacts (reports, quizzes, podcasts) to disk
  - List all artifacts for a notebook (by type)
  - Retrieve a specific artifact's content or file path
  - Directory layout:
      artifacts/reports/*.md
      artifacts/quizzes/*.md
      artifacts/podcasts/*.md + *.mp3
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage.notebook_store import get_artifact_dir, touch_notebook
from utils.security import safe_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TYPES: frozenset[str] = frozenset({"reports", "quizzes", "podcasts"})

# Permitted extensions per artifact type
_ALLOWED_EXT: dict[str, frozenset[str]] = {
    "reports":  frozenset({".md"}),
    "quizzes":  frozenset({".md"}),
    "podcasts": frozenset({".md", ".mp3"}),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_type(artifact_type: str) -> None:
    if artifact_type not in VALID_TYPES:
        raise ValueError(
            f"artifact_type must be one of {set(VALID_TYPES)}, "
            f"got '{artifact_type}'"
        )


def _validate_filename(artifact_type: str, filename: str) -> str:
    """
    Strip directory components and assert the extension is permitted.

    Returns the sanitized bare filename.
    """
    filename = Path(filename).name          # strip any injected path prefix
    ext = Path(filename).suffix.lower()
    allowed = _ALLOWED_EXT[artifact_type]
    if ext not in allowed:
        raise ValueError(
            f"Extension '{ext}' is not allowed for '{artifact_type}'. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return filename


def _resolve(artifact_type: str, username: str, notebook_id: str, filename: str) -> Path:
    """Resolve the full path for an artifact file, guarding traversal."""
    directory = get_artifact_dir(username, notebook_id, artifact_type)
    directory.mkdir(parents=True, exist_ok=True)
    return safe_path(directory, filename)


# ===========================================================================
# Public API  —  signatures match the stub provided in the project brief
# ===========================================================================

def save_artifact(
    username: str,
    notebook_id: str,
    artifact_type: str,
    content: "str | bytes",
    filename: str,
) -> str:
    """
    Save an artifact to disk and return its absolute path.

    Args:
        username      – HuggingFace username (scopes the storage path)
        notebook_id   – UUID of the target notebook
        artifact_type – ``"reports"``, ``"quizzes"``, or ``"podcasts"``
        content       – ``str`` for ``.md`` files; ``bytes`` for ``.mp3`` files
        filename      – desired filename, e.g. ``"report_1.md"``

    Returns:
        Absolute path (str) of the written file.

    Raises:
        ValueError        – invalid *artifact_type* or file extension
        OSError           – underlying filesystem failure

    Example::

        path = save_artifact("alice", nb_id, "reports", markdown_text, "report_1.md")
        path = save_artifact("alice", nb_id, "podcasts", mp3_bytes,     "podcast_1.mp3")
    """
    _validate_type(artifact_type)
    filename = _validate_filename(artifact_type, filename)
    file_path = _resolve(artifact_type, username, notebook_id, filename)

    if isinstance(content, bytes):
        file_path.write_bytes(content)
    else:
        file_path.write_text(content, encoding="utf-8")

    # Keep the notebook's updated_at timestamp current
    try:
        touch_notebook(username, notebook_id)
    except KeyError:
        pass   # notebook deleted mid-session — not a storage error

    return str(file_path)


def list_artifacts(
    username: str,
    notebook_id: str,
    artifact_type: str = None,
) -> list[dict]:
    """
    List artifacts for a notebook, optionally filtered by type.

    Args:
        username      – HuggingFace username
        notebook_id   – UUID of the notebook
        artifact_type – if given, return only this type;
                        if ``None``, return all types combined.

    Returns:
        List of dicts sorted by type → filename::

            {
                "type":       str,   # "reports" | "quizzes" | "podcasts"
                "filename":   str,   # e.g. "report_2.md"
                "path":       str,   # absolute path on disk
                "size":       int,   # bytes
                "created_at": str,   # ISO 8601 UTC (file ctime)
            }

    Raises:
        ValueError – unknown *artifact_type*
    """
    if artifact_type is not None:
        _validate_type(artifact_type)

    types_to_scan = [artifact_type] if artifact_type else sorted(VALID_TYPES)
    results: list[dict] = []

    for atype in types_to_scan:
        try:
            directory = get_artifact_dir(username, notebook_id, atype)
        except ValueError:
            continue
        if not directory.exists():
            continue

        allowed = _ALLOWED_EXT[atype]
        for file_path in sorted(directory.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed:
                continue  # skip .DS_Store and other stray files

            stat = file_path.stat()
            results.append({
                "type":       atype,
                "filename":   file_path.name,
                "path":       str(file_path),
                "size":       stat.st_size,
                "created_at": datetime.fromtimestamp(
                    stat.st_ctime, tz=timezone.utc
                ).isoformat(),
            })

    return results


def get_artifact(
    username: str,
    notebook_id: str,
    artifact_type: str,
    filename: str,
) -> str:
    """
    Read and return an artifact's content as a string.

    For ``.md`` files the UTF-8 text is returned directly.
    For ``.mp3`` files the bytes are decoded as latin-1 (a lossless
    round-trip for arbitrary binary data) so the return type is always
    ``str``.  Use :func:`get_artifact_bytes` when you need real ``bytes``
    (e.g. for Gradio's ``gr.Audio`` component).

    Args:
        username      – HuggingFace username
        notebook_id   – UUID of the notebook
        artifact_type – ``"reports"``, ``"quizzes"``, or ``"podcasts"``
        filename      – bare filename, e.g. ``"report_1.md"``

    Returns:
        File contents as a ``str``.

    Raises:
        FileNotFoundError – artifact does not exist on disk
        ValueError        – invalid *artifact_type* or extension
    """
    _validate_type(artifact_type)
    filename = _validate_filename(artifact_type, filename)
    file_path = _resolve(artifact_type, username, notebook_id, filename)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_type}/{filename} "
            f"(notebook={notebook_id}, user={username})"
        )

    if file_path.suffix.lower() == ".mp3":
        return file_path.read_bytes().decode("latin-1")

    return file_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Extra helpers used by core/artifacts.py and the Gradio UI
# ---------------------------------------------------------------------------

def get_artifact_bytes(
    username: str,
    notebook_id: str,
    artifact_type: str,
    filename: str,
) -> bytes:
    """
    Return an artifact's raw bytes.

    Preferred over :func:`get_artifact` when streaming binary content such
    as podcast audio to Gradio's ``gr.Audio`` component.

    Raises:
        FileNotFoundError – artifact does not exist
        ValueError        – invalid *artifact_type* or extension
    """
    _validate_type(artifact_type)
    filename = _validate_filename(artifact_type, filename)
    file_path = _resolve(artifact_type, username, notebook_id, filename)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_type}/{filename} "
            f"(notebook={notebook_id}, user={username})"
        )

    return file_path.read_bytes()


def delete_artifact(
    username: str,
    notebook_id: str,
    artifact_type: str,
    filename: str,
) -> bool:
    """
    Delete one artifact file from disk.

    Returns:
        ``True``  – file existed and was removed.
        ``False`` – file was not found (no-op, not an error).

    Raises:
        ValueError – invalid *artifact_type* or extension
    """
    _validate_type(artifact_type)
    filename = _validate_filename(artifact_type, filename)
    file_path = _resolve(artifact_type, username, notebook_id, filename)

    if not file_path.exists():
        return False

    file_path.unlink()
    return True