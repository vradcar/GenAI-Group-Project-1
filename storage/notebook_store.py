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


def create_notebook(username: str, name: str) -> dict:
    """Create a new notebook and return its metadata."""
    # TODO
    pass


def list_notebooks(username: str) -> list[dict]:
    """Return all notebooks for a user."""
    # TODO
    pass


def get_notebook(username: str, notebook_id: str) -> dict:
    """Return metadata for a single notebook."""
    # TODO
    pass


def delete_notebook(username: str, notebook_id: str) -> None:
    """Delete a notebook and all its data."""
    # TODO
    pass
