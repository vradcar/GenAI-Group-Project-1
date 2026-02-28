"""Tests for utils/extractors.py"""

from unittest.mock import MagicMock, patch

import pytest

from utils.extractors import extract_pdf, extract_pptx, extract_txt, extract_url


# extract_txt  

def test_extract_txt_returns_content(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello, StudyPod!", encoding="utf-8")
    assert extract_txt(str(f)) == "Hello, StudyPod!"


def test_extract_txt_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert extract_txt(str(f)) == ""


def test_extract_txt_replaces_bad_encoding(tmp_path):
    f = tmp_path / "latin.txt"
    f.write_bytes(b"caf\xe9")  # invalid UTF-8 byte
    result = extract_txt(str(f))
    assert "caf" in result  # bad byte replaced, rest kept


def test_extract_txt_multiline(tmp_path):
    content = "line one\nline two\nline three"
    f = tmp_path / "multi.txt"
    f.write_text(content, encoding="utf-8")
    assert extract_txt(str(f)) == content


# extract_pdf

def test_extract_pdf_returns_text():
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page one content.\n"

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter([mock_page])

    with patch("fitz.open", return_value=mock_doc):
        result = extract_pdf("dummy.pdf")

    assert "Page one content." in result
    mock_doc.close.assert_called_once()


def test_extract_pdf_skips_blank_pages():
    blank_page = MagicMock()
    blank_page.get_text.return_value = "   \n"
    text_page = MagicMock()
    text_page.get_text.return_value = "Real content.\n"

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter([blank_page, text_page])

    with patch("fitz.open", return_value=mock_doc):
        result = extract_pdf("dummy.pdf")

    assert "Real content." in result
    assert result.strip() == "Real content."


def test_extract_pdf_multiple_pages():
    pages = []
    for i in range(3):
        p = MagicMock()
        p.get_text.return_value = f"Page {i + 1} text.\n"
        pages.append(p)

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter(pages)

    with patch("fitz.open", return_value=mock_doc):
        result = extract_pdf("multi.pdf")

    for i in range(1, 4):
        assert f"Page {i} text." in result


# extract_pptx 

def _make_shape(text: str) -> MagicMock:
    shape = MagicMock()
    shape.text = text
    return shape


def test_extract_pptx_returns_text():
    shape1 = _make_shape("Title slide")
    shape2 = _make_shape("Bullet point one")

    mock_slide = MagicMock()
    mock_slide.shapes = [shape1, shape2]

    mock_prs = MagicMock()
    mock_prs.slides = [mock_slide]

    with patch("pptx.Presentation", return_value=mock_prs):
        result = extract_pptx("deck.pptx")

    assert "Title slide" in result
    assert "Bullet point one" in result


def test_extract_pptx_multiple_slides():
    slides = []
    for i in range(3):
        shape = _make_shape(f"Slide {i + 1} content")
        slide = MagicMock()
        slide.shapes = [shape]
        slides.append(slide)

    mock_prs = MagicMock()
    mock_prs.slides = slides

    with patch("pptx.Presentation", return_value=mock_prs):
        result = extract_pptx("multi.pptx")

    for i in range(1, 4):
        assert f"Slide {i} content" in result


def test_extract_pptx_skips_blank_shapes():
    blank = _make_shape("   ")
    real = _make_shape("Non-blank text")

    mock_slide = MagicMock()
    mock_slide.shapes = [blank, real]

    mock_prs = MagicMock()
    mock_prs.slides = [mock_slide]

    with patch("pptx.Presentation", return_value=mock_prs):
        result = extract_pptx("deck.pptx")

    assert "Non-blank text" in result
    assert result.strip() == "Non-blank text"


# extract_url 

def test_extract_url_success():
    with patch("trafilatura.fetch_url", return_value="<html>content</html>"), \
         patch("trafilatura.extract", return_value="Extracted web article"):
        result = extract_url("https://example.com/article")
    assert result == "Extracted web article"


def test_extract_url_fetch_fails():
    with patch("trafilatura.fetch_url", return_value=None):
        with pytest.raises(ValueError, match="Could not fetch"):
            extract_url("https://unreachable.example.com")


def test_extract_url_extract_returns_none():
    with patch("trafilatura.fetch_url", return_value="<html></html>"), \
         patch("trafilatura.extract", return_value=None):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_url("https://empty-page.example.com")


def test_extract_url_passes_url_to_fetch():
    target = "https://example.com/news/story"
    with patch("trafilatura.fetch_url", return_value="<html>x</html>") as mock_fetch, \
         patch("trafilatura.extract", return_value="Story text"):
        extract_url(target)
    mock_fetch.assert_called_once_with(target)
