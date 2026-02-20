"""
Text Extractors — file-type-specific text extraction.

Responsibilities:
  - extract_pdf(path)  → str   (via PyMuPDF)
  - extract_pptx(path) → str   (via python-pptx)
  - extract_txt(path)  → str   (plain read)
  - extract_url(url)   → str   (via trafilatura)
"""


def extract_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    # TODO
    pass


def extract_pptx(file_path: str) -> str:
    """Extract text from a PPTX file using python-pptx."""
    # TODO
    pass


def extract_txt(file_path: str) -> str:
    """Read and return the contents of a plain text file."""
    # TODO
    pass


def extract_url(url: str) -> str:
    """Fetch a URL and extract the main content using trafilatura."""
    # TODO
    pass
