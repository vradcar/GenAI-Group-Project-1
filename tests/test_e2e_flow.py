"""
End-to-end integration test: ingest → vector store → RAG.

Uses real embeddings (sentence-transformers) and real ChromaDB on disk,
but mocks the Groq LLM since no API key is available in CI.
"""

import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.models import LLMResponse

# ---------------------------------------------------------------------------
# Sample document — long enough to produce multiple chunks
# ---------------------------------------------------------------------------

SAMPLE_DOC = textwrap.dedent("""\
    The Theory of Relativity

    Albert Einstein published the special theory of relativity in 1905. It
    introduced the famous equation E=mc^2, which describes the relationship
    between energy and mass. The theory fundamentally changed our understanding
    of space and time, showing they are interwoven into a single continuum
    known as spacetime.

    General relativity, published in 1915, extended special relativity to include
    gravity. Einstein proposed that massive objects cause a distortion in
    spacetime, which is felt as gravity. This was confirmed during the 1919
    solar eclipse when starlight was observed bending around the sun, exactly
    as Einstein predicted.

    Quantum mechanics, developed in the early 20th century, provides a
    mathematical description of the behavior of subatomic particles. Unlike
    classical mechanics, quantum mechanics is inherently probabilistic. The
    Heisenberg uncertainty principle states that one cannot simultaneously
    know both the exact position and exact momentum of a particle.

    The Standard Model of particle physics classifies all known elementary
    particles. It includes quarks, leptons, gauge bosons, and the Higgs boson,
    discovered at CERN in 2012. The model has been remarkably successful in
    predicting experimental results, but it does not include gravity.

    String theory attempts to reconcile general relativity and quantum mechanics.
    It proposes that the fundamental constituents of the universe are not
    point-like particles but tiny vibrating strings of energy. Different
    vibrational modes of these strings correspond to different particles.
    String theory predicts the existence of extra spatial dimensions beyond
    the three we observe.
""")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_root(tmp_path):
    """Redirect DATA_ROOT to a temp directory for isolation."""
    return tmp_path / "data"


@pytest.fixture()
def notebook_env(data_root):
    """
    Patch DATA_ROOT across all modules that import it, so every path
    helper (notebook_store, vector_store, ingestion) writes to tmp_path.
    """
    patches = [
        patch("utils.config.DATA_ROOT", data_root),
        patch("storage.notebook_store.DATA_ROOT", data_root),
    ]
    for p in patches:
        p.start()
    yield {"data_root": data_root, "username": "testuser", "notebook_id": "e2e-nb-001"}
    for p in patches:
        p.stop()


@pytest.fixture()
def txt_file(tmp_path):
    """Write the sample document to a .txt file and return its path."""
    p = tmp_path / "physics.txt"
    p.write_text(SAMPLE_DOC, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline: ingest file → query vector store → RAG answer."""

    def test_ingest_creates_chunks_in_chromadb(self, notebook_env, txt_file):
        """After ingesting a .txt file, ChromaDB should contain chunks."""
        from core.ingestion import ingest_file
        from storage import vector_store

        env = notebook_env
        result = ingest_file(env["username"], env["notebook_id"], str(txt_file))

        # ingest_file should return success
        assert result["status"] == "ok"
        assert result["source"] == "physics.txt"
        assert result["chunks"] >= 2  # long enough for >1 chunk
        assert result["extracted_chars"] > 0

        # ChromaDB should have the chunks
        count = vector_store.collection_count(env["username"], env["notebook_id"])
        assert count == result["chunks"]

    def test_vector_store_query_returns_relevant_chunks(self, notebook_env, txt_file):
        """Querying the vector store after ingestion should return relevant results."""
        from core.ingestion import ingest_file
        from storage import vector_store

        env = notebook_env
        ingest_file(env["username"], env["notebook_id"], str(txt_file))

        # Query about Einstein — should match the relativity chunks
        results = vector_store.query_collection(
            env["username"], env["notebook_id"],
            "What did Einstein publish?",
            n_results=3,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        assert len(docs) >= 1
        assert len(docs) == len(metas) == len(distances)
        # At least one chunk should mention Einstein or relativity
        combined = " ".join(docs).lower()
        assert "einstein" in combined or "relativity" in combined

    def test_list_sources_after_ingest(self, notebook_env, txt_file):
        """list_sources should show the ingested file."""
        from core.ingestion import ingest_file
        from storage import vector_store

        env = notebook_env
        ingest_file(env["username"], env["notebook_id"], str(txt_file))

        sources = vector_store.list_sources(env["username"], env["notebook_id"])
        assert "physics.txt" in sources

    def test_rag_query_naive_returns_answer_with_citations(self, notebook_env, txt_file):
        """
        Full RAG flow: ingest → retrieve → LLM (mocked) → RAGResponse.

        Only the Groq API is mocked. Embeddings + ChromaDB are real.
        """
        from core.ingestion import ingest_file
        from core.rag import query as rag_query

        env = notebook_env
        ingest_file(env["username"], env["notebook_id"], str(txt_file))

        fake_llm = LLMResponse(
            text="Einstein published the special theory of relativity in 1905 [1].",
            model="fake-model",
            usage={"total_tokens": 42},
        )

        with patch("core.rag.llm_client.complete", return_value=fake_llm):
            response = rag_query(
                env["username"], env["notebook_id"],
                "What did Einstein publish?",
                technique="naive",
            )

        assert response.answer == fake_llm.text
        assert response.technique == "naive"
        assert response.chunks_considered >= 1
        assert len(response.citations) >= 1
        # Each citation should have meaningful fields
        for cit in response.citations:
            assert cit.chunk_text  # non-empty
            assert 0.0 <= cit.relevance_score <= 1.0

    def test_rag_query_empty_notebook_returns_no_docs_message(self, notebook_env):
        """RAG on an empty notebook should return a helpful message, not crash."""
        from core.rag import query as rag_query

        env = notebook_env
        response = rag_query(
            env["username"], env["notebook_id"],
            "What is physics?",
            technique="naive",
        )

        assert "no documents" in response.answer.lower() or response.chunks_considered == 0
        assert response.citations == []

    def test_idempotent_ingest(self, notebook_env, txt_file):
        """Ingesting the same file twice should not duplicate chunks."""
        from core.ingestion import ingest_file
        from storage import vector_store

        env = notebook_env
        r1 = ingest_file(env["username"], env["notebook_id"], str(txt_file))
        r2 = ingest_file(env["username"], env["notebook_id"], str(txt_file))

        assert r1["chunks"] == r2["chunks"]
        count = vector_store.collection_count(env["username"], env["notebook_id"])
        assert count == r1["chunks"]  # no duplicates

    def test_delete_source_removes_chunks(self, notebook_env, txt_file):
        """Deleting a source should remove its chunks from the vector store."""
        from core.ingestion import ingest_file
        from storage import vector_store

        env = notebook_env
        result = ingest_file(env["username"], env["notebook_id"], str(txt_file))
        assert vector_store.collection_count(env["username"], env["notebook_id"]) > 0

        deleted = vector_store.delete_source(
            env["username"], env["notebook_id"], "physics.txt"
        )
        assert deleted == result["chunks"]
        assert vector_store.collection_count(env["username"], env["notebook_id"]) == 0
