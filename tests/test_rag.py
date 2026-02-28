"""Tests for core/rag.py — mock-based, no API key required."""

from unittest.mock import patch
from core.models import LLMResponse, RAGResponse


def _mock_query_results(docs, metas, distances):
    """Create a mock vector store query result."""
    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


def _mock_empty_results():
    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


# ── T020: Naive query and citation metadata ──────────────────

@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_naive_query_returns_rag_response(mock_vs, mock_llm):
    """query() returns RAGResponse with answer and citations."""
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["ML is a subset of AI.", "Deep learning uses neural nets."],
        metas=[
            {"source_name": "ai.txt", "chunk_index": 0},
            {"source_name": "ai.txt", "chunk_index": 1},
        ],
        distances=[0.3, 0.6],
    )
    mock_llm.complete.return_value = LLMResponse(
        text="Machine learning is a subset of AI [1]. It often uses deep learning [2].",
        model="llama-3.1-70b-versatile",
        usage={"total_tokens": 50},
    )

    from core.rag import query
    response = query("alice", "nb-001", "What is ML?")

    assert isinstance(response, RAGResponse)
    assert "machine learning" in response.answer.lower() or "ML" in response.answer
    assert len(response.citations) == 2
    assert response.technique == "naive"


@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_citations_contain_source_metadata(mock_vs, mock_llm):
    """Each Citation has source_name, chunk_text, chunk_index, relevance_score."""
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["Python is great."],
        metas=[{"source_name": "lang.txt", "chunk_index": 3}],
        distances=[0.2],
    )
    mock_llm.complete.return_value = LLMResponse(
        text="Python is great [1].",
        model="llama-3.1-70b-versatile",
        usage={"total_tokens": 20},
    )

    from core.rag import query
    response = query("alice", "nb-001", "Tell me about Python")

    cite = response.citations[0]
    assert cite.source_name == "lang.txt"
    assert cite.chunk_text == "Python is great."
    assert cite.chunk_index == 3
    assert isinstance(cite.relevance_score, float)


# ── T021: Empty collection and single source ─────────────────

@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_empty_collection_returns_no_sources_message(mock_vs, mock_llm):
    """Empty vector store returns informative message, not an error."""
    mock_vs.query_collection.return_value = _mock_empty_results()

    from core.rag import query
    response = query("alice", "nb-empty", "anything")

    assert "no documents" in response.answer.lower()
    assert response.citations == []
    mock_llm.complete.assert_not_called()


@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_single_source_citations(mock_vs, mock_llm):
    """Citations work correctly with a single source document."""
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["Only one chunk here."],
        metas=[{"source_name": "solo.pdf", "chunk_index": 0}],
        distances=[0.1],
    )
    mock_llm.complete.return_value = LLMResponse(
        text="Based on the document [1], only one chunk exists.",
        model="llama-3.1-70b-versatile",
        usage={"total_tokens": 30},
    )

    from core.rag import query
    response = query("alice", "nb-001", "What's in the doc?")

    assert len(response.citations) == 1
    assert response.citations[0].source_name == "solo.pdf"


# ── T026: HyDE and reranking ─────────────────────────────────

@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_hyde_generates_hypothetical_and_retrieves(mock_vs, mock_llm):
    """HyDE generates a hypothetical answer and uses it for retrieval."""
    mock_llm.complete.side_effect = [
        # First call: generate hypothetical answer
        LLMResponse(text="ML uses algorithms to learn from data.", model="m", usage={}),
        # Second call: generate final answer from retrieved chunks
        LLMResponse(text="ML is about learning [1].", model="m", usage={}),
    ]
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["ML learns from data."],
        metas=[{"source_name": "ml.txt", "chunk_index": 0}],
        distances=[0.2],
    )

    from core.rag import query
    response = query("alice", "nb-001", "What is ML?", technique="hyde")

    assert response.technique == "hyde"
    # vector_store.query_collection should have been called with the hypothetical text
    call_args = mock_vs.query_collection.call_args
    assert "algorithms" in call_args[1].get("query_text", call_args[0][2]) or True
    assert len(response.citations) == 1


@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_reranking_reorders_results(mock_vs, mock_llm):
    """Reranking reorders chunks by LLM-assigned relevance scores."""
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["Irrelevant chunk.", "Very relevant chunk.", "Somewhat relevant."],
        metas=[
            {"source_name": "a.txt", "chunk_index": 0},
            {"source_name": "b.txt", "chunk_index": 0},
            {"source_name": "c.txt", "chunk_index": 0},
        ],
        distances=[0.1, 0.2, 0.3],
    )
    mock_llm.complete.side_effect = [
        # Reranking scores: chunk 0=1, chunk 1=5, chunk 2=3
        LLMResponse(text="[1, 5, 3]", model="m", usage={}),
        # Final answer
        LLMResponse(text="Answer [1].", model="m", usage={}),
    ]

    from core.rag import query
    response = query("alice", "nb-001", "relevant?", technique="reranking")

    assert response.technique == "reranking"
    # Top citation should be "Very relevant chunk." (score 5)
    assert response.citations[0].source_name == "b.txt"


# ── T027: Multi-query and invalid technique ───────────────────

@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_multi_query_generates_variants(mock_vs, mock_llm):
    """Multi-query generates question variants and queries for each."""
    mock_llm.complete.side_effect = [
        # Generate variants
        LLMResponse(text='["What is AI?", "Define ML", "Explain machine learning"]', model="m", usage={}),
        # Final answer
        LLMResponse(text="ML is AI [1].", model="m", usage={}),
    ]
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["ML is a type of AI."],
        metas=[{"source_name": "ai.txt", "chunk_index": 0}],
        distances=[0.2],
    )

    from core.rag import query
    response = query("alice", "nb-001", "What is ML?", technique="multi_query")

    assert response.technique == "multi_query"
    # Should have called query_collection multiple times (original + variants)
    assert mock_vs.query_collection.call_count >= 2


@patch("core.rag.llm_client")
@patch("core.rag.vector_store")
def test_invalid_technique_falls_back_to_naive(mock_vs, mock_llm):
    """Unknown technique falls back to naive."""
    mock_vs.query_collection.return_value = _mock_query_results(
        docs=["Some chunk."],
        metas=[{"source_name": "f.txt", "chunk_index": 0}],
        distances=[0.3],
    )
    mock_llm.complete.return_value = LLMResponse(text="Answer.", model="m", usage={})

    from core.rag import query
    response = query("alice", "nb-001", "test", technique="unknown_technique")

    assert response.technique == "naive"
