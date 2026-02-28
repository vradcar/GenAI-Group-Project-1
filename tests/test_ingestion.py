"""Tests for core/ingestion.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.ingestion import ingest_file, ingest_url

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_TEXT = "This is a chunk of meaningful content. " * 40  # > 1000 chars


def _make_dirs(tmp_path, username="user", notebook_id="nb1"):
    """Create the notebook directory structure that notebook_store would create."""
    nb_dir = tmp_path / username / notebook_id
    raw_dir = nb_dir / "files_raw"
    extracted_dir = nb_dir / "files_extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    return nb_dir, raw_dir, extracted_dir


def _patch_deps(
    *,
    tmp_path,
    username: str = "user",
    notebook_id: str = "nb1",
    safe_name: str = "doc.txt",
    extracted_text: str = LONG_TEXT,
    extractor_fn: str = "utils.extractors.extract_txt",
):
    """Return a tuple of context managers that mock external dependencies."""
    nb_dir = tmp_path / username / notebook_id
    raw_dir = nb_dir / "files_raw"
    extracted_dir = nb_dir / "files_extracted"
    return (
        patch("core.ingestion.get_notebook_dir", return_value=nb_dir),
        patch("core.ingestion.get_raw_dir", return_value=raw_dir),
        patch("core.ingestion.get_extracted_dir", return_value=extracted_dir),
        patch("core.ingestion.sanitize_filename", return_value=safe_name),
        patch("core.ingestion.validate_path", side_effect=lambda p, _: Path(p)),
        patch(extractor_fn, return_value=extracted_text),
        patch("storage.vector_store.add_documents"),
    )


# ---------------------------------------------------------------------------
# ingest_file — validation
# ---------------------------------------------------------------------------

def test_ingest_file_rejects_invalid_extension(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_file("user", "nb1", str(f))


def test_ingest_file_rejects_oversized_file(tmp_path):
    f = tmp_path / "big.pdf"
    f.write_bytes(b"x")
    stat_mock = MagicMock()
    stat_mock.st_size = 51 * 1024 * 1024  # 51 MB
    with patch.object(Path, "stat", return_value=stat_mock):
        with pytest.raises(ValueError, match="exceeds"):
            ingest_file("user", "nb1", str(f))


def test_ingest_file_rejects_empty_extraction(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("   ")
    _make_dirs(tmp_path)
    patches = _patch_deps(tmp_path=tmp_path, extracted_text="   ")
    # Override extractor to return blank
    patched = list(patches)
    patched[5] = patch("utils.extractors.extract_txt", return_value="   ")
    with patched[0], patched[1], patched[2], patched[3], patched[4], patched[5], patched[6]:
        with pytest.raises(ValueError, match="No text"):
            ingest_file("user", "nb1", str(f))


# ---------------------------------------------------------------------------
# ingest_file — happy path (one test per allowed extension)
# ---------------------------------------------------------------------------

def test_ingest_file_txt_returns_ok_dict(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(LONG_TEXT, encoding="utf-8")

    _make_dirs(tmp_path)
    p1, p2, p3, p4, p5, p6, p7 = _patch_deps(tmp_path=tmp_path)
    with p1, p2, p3, p4, p5, p6 as mock_extract, p7 as mock_add:
        result = ingest_file("user", "nb1", str(f))

    assert result["status"] == "ok"
    assert result["source"] == "doc.txt"
    assert result["chunks"] >= 1
    assert result["extracted_chars"] == len(LONG_TEXT)
    mock_extract.assert_called_once()
    mock_add.assert_called_once()


def test_ingest_file_pdf(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF fake")

    _make_dirs(tmp_path)
    p1, p2, p3, p4, p5, p6, p7 = _patch_deps(
        tmp_path=tmp_path,
        safe_name="report.pdf",
        extractor_fn="utils.extractors.extract_pdf",
    )
    with p1, p2, p3, p4, p5, p6, p7 as mock_add:
        result = ingest_file("user", "nb1", str(f))

    assert result["status"] == "ok"
    mock_add.assert_called_once()


def test_ingest_file_pptx(tmp_path):
    f = tmp_path / "slides.pptx"
    f.write_bytes(b"PK fake")

    _make_dirs(tmp_path)
    p1, p2, p3, p4, p5, p6, p7 = _patch_deps(
        tmp_path=tmp_path,
        safe_name="slides.pptx",
        extractor_fn="utils.extractors.extract_pptx",
    )
    with p1, p2, p3, p4, p5, p6, p7 as mock_add:
        result = ingest_file("user", "nb1", str(f))

    assert result["status"] == "ok"
    mock_add.assert_called_once()


# ---------------------------------------------------------------------------
# ingest_file — vector store contract
# ---------------------------------------------------------------------------

def test_ingest_file_passes_correct_metadata_to_vector_store(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text(LONG_TEXT, encoding="utf-8")

    _make_dirs(tmp_path, "alice", "nb-42")
    p1, p2, p3, p4, p5, p6, p7 = _patch_deps(tmp_path=tmp_path, username="alice", notebook_id="nb-42")
    with p1, p2, p3, p4, p5, p6, p7 as mock_add:
        ingest_file("alice", "nb-42", str(f))

    args = mock_add.call_args
    username_arg, notebook_id_arg, chunks_arg, metadatas_arg = args[0]

    assert username_arg == "alice"
    assert notebook_id_arg == "nb-42"
    assert isinstance(chunks_arg, list) and len(chunks_arg) >= 1
    for i, meta in enumerate(metadatas_arg):
        assert meta["source"] == "doc.txt"
        assert meta["chunk_index"] == i
        assert meta["source_type"] == "file"


# ---------------------------------------------------------------------------
# ingest_file — directory creation
# ---------------------------------------------------------------------------

def test_ingest_file_creates_raw_and_extracted_dirs(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text(LONG_TEXT, encoding="utf-8")

    _make_dirs(tmp_path)
    p1, p2, p3, p4, p5, p6, p7 = _patch_deps(tmp_path=tmp_path)
    with p1, p2, p3, p4, p5, p6, p7:
        ingest_file("user", "nb1", str(f))

    nb_dir = tmp_path / "user" / "nb1"
    assert (nb_dir / "files_raw").is_dir()
    assert (nb_dir / "files_extracted").is_dir()


# ---------------------------------------------------------------------------
# ingest_url — validation
# ---------------------------------------------------------------------------

def test_ingest_url_raises_on_empty_extraction(tmp_path):
    _make_dirs(tmp_path)
    nb_dir = tmp_path / "user" / "nb1"
    with patch("core.ingestion.get_notebook_dir", return_value=nb_dir), \
         patch("core.ingestion.get_extracted_dir", return_value=nb_dir / "files_extracted"), \
         patch("utils.security.sanitize_filename", return_value="page"), \
         patch("utils.security.validate_path", side_effect=lambda p, _: Path(p)), \
         patch("utils.extractors.extract_url", return_value="   "):
        with pytest.raises(ValueError, match="No text"):
            ingest_url("user", "nb1", "https://example.com")


def test_ingest_url_propagates_extractor_error():
    with patch("utils.extractors.extract_url", side_effect=ValueError("Could not fetch")):
        with pytest.raises(ValueError, match="Could not fetch"):
            ingest_url("user", "nb1", "https://broken.example.com")


# ---------------------------------------------------------------------------
# ingest_url — happy path
# ---------------------------------------------------------------------------

def test_ingest_url_returns_ok_dict(tmp_path):
    _make_dirs(tmp_path)
    nb_dir = tmp_path / "user" / "nb1"
    with patch("core.ingestion.get_notebook_dir", return_value=nb_dir), \
         patch("core.ingestion.get_extracted_dir", return_value=nb_dir / "files_extracted"), \
         patch("utils.security.sanitize_filename", return_value="article"), \
         patch("utils.security.validate_path", side_effect=lambda p, _: Path(p)), \
         patch("utils.extractors.extract_url", return_value=LONG_TEXT), \
         patch("storage.vector_store.add_documents") as mock_add:
        result = ingest_url("user", "nb1", "https://example.com/article")

    assert result["status"] == "ok"
    assert result["source"] == "https://example.com/article"
    assert result["chunks"] >= 1
    assert result["extracted_chars"] == len(LONG_TEXT)
    mock_add.assert_called_once()


def test_ingest_url_passes_correct_metadata_to_vector_store(tmp_path):
    url = "https://example.com/news"
    _make_dirs(tmp_path, "bob", "nb-99")
    nb_dir = tmp_path / "bob" / "nb-99"
    with patch("core.ingestion.get_notebook_dir", return_value=nb_dir), \
         patch("core.ingestion.get_extracted_dir", return_value=nb_dir / "files_extracted"), \
         patch("utils.security.sanitize_filename", return_value="news"), \
         patch("utils.security.validate_path", side_effect=lambda p, _: Path(p)), \
         patch("utils.extractors.extract_url", return_value=LONG_TEXT), \
         patch("storage.vector_store.add_documents") as mock_add:
        ingest_url("bob", "nb-99", url)

    username_arg, notebook_id_arg, _, metadatas_arg = mock_add.call_args[0]
    assert username_arg == "bob"
    assert notebook_id_arg == "nb-99"
    for i, meta in enumerate(metadatas_arg):
        assert meta["source"] == url
        assert meta["chunk_index"] == i
        assert meta["source_type"] == "url"


def test_ingest_url_creates_extracted_dir(tmp_path):
    _make_dirs(tmp_path)
    nb_dir = tmp_path / "user" / "nb1"
    with patch("core.ingestion.get_notebook_dir", return_value=nb_dir), \
         patch("core.ingestion.get_extracted_dir", return_value=nb_dir / "files_extracted"), \
         patch("utils.security.sanitize_filename", return_value="page"), \
         patch("utils.security.validate_path", side_effect=lambda p, _: Path(p)), \
         patch("utils.extractors.extract_url", return_value=LONG_TEXT), \
         patch("storage.vector_store.add_documents"):
        ingest_url("user", "nb1", "https://example.com")

    assert (tmp_path / "user" / "nb1" / "files_extracted").is_dir()
