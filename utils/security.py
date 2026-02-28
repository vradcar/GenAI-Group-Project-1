"""
Security utilities.

Responsibilities:
  - Sanitize filenames (strip ../, /, \\, null bytes)
  - Sanitize notebook names (alphanumeric + spaces + hyphens, max 100 chars)
  - Sanitize usernames
  - Validate that resolved paths stay inside the user's directory
  - Generate UUID4 notebook IDs
"""

import re
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Maximum lengths
_MAX_FILENAME_LEN      = 255   # hard filesystem limit on most OSes
_MAX_NOTEBOOK_NAME_LEN = 100   # specified in the docstring
_MAX_USERNAME_LEN      = 64    # generous but bounded

# Notebook names: alphanumeric, spaces, hyphens only (per docstring)
_NOTEBOOK_NAME_ALLOWED = re.compile(r"[^a-zA-Z0-9 \-]")

# Usernames: alphanumeric and hyphens (HuggingFace convention)
_USERNAME_ALLOWED = re.compile(r"[^a-zA-Z0-9\-]")

# Characters that are dangerous in filenames on any major OS
_FILENAME_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')

# Windows reserved filenames (blocked everywhere for portability)
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# ===========================================================================
# Public API  —  signatures match the stub exactly
# ===========================================================================

def sanitize_filename(name: str) -> str:
    """
    Remove dangerous characters from a filename.

    Specifically strips:
      - Path traversal sequences  (../, ..\\ and any remaining / or \\)
      - Null bytes and ASCII control characters
      - OS-unsafe characters      (< > : " | ? *)
      - Leading and trailing dots and spaces (hidden-file / Windows quirk)

    Also enforces a maximum length of 255 characters (preserving extension).

    Args:
        name – raw filename from user input or an uploaded file object

    Returns:
        Sanitized filename safe to write to disk.

    Raises:
        ValueError – result is empty after sanitization
    """
    if not name:
        raise ValueError("Filename must not be empty.")

    # 1. Take only the basename — strips any directory prefix the caller
    #    included (e.g. "../../etc/passwd" → "passwd")
    name = Path(name).name

    # 2. Remove null bytes and control characters explicitly
    name = name.replace("\x00", "")
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)

    # 3. Replace remaining OS-unsafe characters with underscores
    name = _FILENAME_UNSAFE.sub("_", name)

    # 4. Collapse consecutive dots (prevents "file...exe" extension spoofing)
    name = re.sub(r"\.{2,}", ".", name)

    # 5. Strip leading/trailing dots and spaces
    name = name.strip(". ")

    # 6. Enforce max length (preserve extension)
    if len(name) > _MAX_FILENAME_LEN:
        stem, _, ext = name.rpartition(".")
        ext_part = ("." + ext) if ext else ""
        name = stem[: _MAX_FILENAME_LEN - len(ext_part)] + ext_part

    # 7. Prefix Windows reserved names so they remain usable cross-platform
    stem_upper = name.rsplit(".", 1)[0].upper()
    if stem_upper in _WINDOWS_RESERVED:
        name = "_" + name

    if not name:
        raise ValueError("Filename is empty after sanitization.")

    return name


def sanitize_notebook_name(name: str) -> str:
    """
    Enforce allowed chars and max length on notebook names.

    Allowed characters: alphanumeric, spaces, hyphens  (per docstring spec).
    Maximum length: 100 characters.

    Steps:
      1. Strip leading/trailing whitespace
      2. Remove any character that isn't [a-zA-Z0-9 -]
      3. Collapse multiple consecutive spaces to one
      4. Truncate to 100 characters and strip again

    Args:
        name – raw notebook name from user input

    Returns:
        Sanitized notebook name.

    Raises:
        ValueError – result is empty after sanitization
    """
    if not name or not name.strip():
        raise ValueError("Notebook name must not be blank.")

    # 1. Strip surrounding whitespace
    name = name.strip()

    # 2. Remove disallowed characters (keep alphanumeric, space, hyphen)
    name = _NOTEBOOK_NAME_ALLOWED.sub("", name)

    # 3. Collapse multiple spaces to a single space
    name = re.sub(r" {2,}", " ", name)

    # 4. Truncate and re-strip (truncation might leave a trailing space)
    name = name[:_MAX_NOTEBOOK_NAME_LEN].strip()

    if not name:
        raise ValueError(
            "Notebook name is empty after sanitization. "
            "Use alphanumeric characters, spaces, or hyphens."
        )

    return name


def sanitize_username(name: str) -> str:
    """
    Clean a username for use as a directory name.

    HuggingFace usernames consist of alphanumeric characters and hyphens.
    We lower-case the result and replace anything outside [a-z0-9-] with
    an underscore, making it safe as a filesystem path component.

    Args:
        name – raw username (typically from HF OAuth profile)

    Returns:
        Lowercase, filesystem-safe username string.

    Raises:
        ValueError – result is empty after sanitization
    """
    if not name or not name.strip():
        raise ValueError("Username must not be empty.")

    # Lower-case first (HF usernames are case-insensitive in practice)
    name = name.strip().lower()

    # Replace anything that isn't alphanumeric or hyphen with underscore
    name = _USERNAME_ALLOWED.sub("_", name)

    # Truncate to maximum length
    name = name[:_MAX_USERNAME_LEN]

    if not name:
        raise ValueError("Username is empty after sanitization.")

    return name


def validate_path(path: str, allowed_root: str) -> Path:
    """
    Resolve a path and verify it's inside allowed_root.

    Both arguments are strings (as specified by the stub).  Resolution is
    done with ``Path.resolve()`` so symlinks, ``.`` and ``..`` components are
    all expanded before the containment check.

    Args:
        path         – candidate path (may be relative or contain ``..``)
        allowed_root – the root directory the resolved path must stay within

    Returns:
        Resolved ``Path`` object, guaranteed to be inside *allowed_root*.

    Raises:
        ValueError – resolved path escapes *allowed_root*
    """
    resolved      = Path(path).resolve()
    resolved_root = Path(allowed_root).resolve()

    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Path '{resolved}' is outside the allowed root '{resolved_root}'."
        )

    return resolved


def generate_notebook_id() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


def safe_path(base: Path, *paths: str) -> Path:
    """
    Join base with paths and validate result stays inside base.
    """
    combined = base.joinpath(*paths)
    return validate_path(str(combined), str(base))