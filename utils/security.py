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


def sanitize_filename(name: str) -> str:
    """Remove dangerous characters from a filename."""
    # TODO
    pass


def sanitize_notebook_name(name: str) -> str:
    """Enforce allowed chars and max length on notebook names."""
    # TODO
    pass


def sanitize_username(name: str) -> str:
    """Clean a username for use as a directory name."""
    # TODO
    pass


def validate_path(path: str, allowed_root: str) -> Path:
    """Resolve a path and verify it's inside allowed_root. Raise ValueError if not."""
    # TODO
    pass


def generate_notebook_id() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())
