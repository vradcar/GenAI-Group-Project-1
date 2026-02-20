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


def save_artifact(username: str, notebook_id: str, artifact_type: str, content: str, filename: str) -> str:
    """Save an artifact to disk and return its path."""
    # TODO
    pass


def list_artifacts(username: str, notebook_id: str, artifact_type: str = None) -> list[dict]:
    """List artifacts, optionally filtered by type."""
    # TODO
    pass


def get_artifact(username: str, notebook_id: str, artifact_type: str, filename: str) -> str:
    """Read and return an artifact's content."""
    # TODO
    pass
