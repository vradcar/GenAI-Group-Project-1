"""Tests for storage/vector_store.py — uses real ChromaDB with temp dirs."""

import tempfile
import pytest
from unittest.mock import patch


@pytest.fixture
def temp_data_dir():
    """Create a temp directory and patch DATA_DIR to use it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("storage.vector_store.DATA_DIR", tmpdir):
            with patch("utils.config.DATA_DIR", tmpdir):
                yield tmpdir


def test_add_and_query_documents(temp_data_dir):
    """Add chunks and query — top result should be semantically closest."""
    from storage.vector_store import add_documents, query_collection

    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Neural networks are inspired by biological brains.",
    ]
    metadatas = [
        {"source_name": "ai.txt", "chunk_index": 0},
        {"source_name": "python.txt", "chunk_index": 0},
        {"source_name": "ai.txt", "chunk_index": 1},
    ]

    add_documents("testuser", "nb-001", chunks, metadatas)
    results = query_collection("testuser", "nb-001", "What is machine learning?", n_results=2)

    assert len(results["documents"][0]) == 2
    assert "machine learning" in results["documents"][0][0].lower() or \
           "artificial intelligence" in results["documents"][0][0].lower()


def test_idempotent_upsert(temp_data_dir):
    """Adding same chunks twice should not create duplicates."""
    from storage.vector_store import add_documents, get_or_create_collection

    chunks = ["Hello world", "Goodbye world"]
    metadatas = [
        {"source_name": "test.txt", "chunk_index": 0},
        {"source_name": "test.txt", "chunk_index": 1},
    ]

    add_documents("testuser", "nb-002", chunks, metadatas)
    add_documents("testuser", "nb-002", chunks, metadatas)

    collection = get_or_create_collection("testuser", "nb-002")
    assert collection.count() == 2


def test_empty_collection_query(temp_data_dir):
    """Querying an empty collection returns empty results, not an error."""
    from storage.vector_store import query_collection

    results = query_collection("testuser", "nb-empty", "anything")

    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]
    assert results["distances"] == [[]]


def test_delete_collection(temp_data_dir):
    """Delete collection removes all data."""
    from storage.vector_store import add_documents, delete_collection, get_or_create_collection

    add_documents("testuser", "nb-del", ["test chunk"],
                  [{"source_name": "f.txt", "chunk_index": 0}])

    delete_collection("testuser", "nb-del")

    collection = get_or_create_collection("testuser", "nb-del")
    assert collection.count() == 0
