"""
RAG Engine.

Responsibilities:
  - Retrieve top-k chunks from ChromaDB by cosine similarity
  - Assemble the prompt with retrieved chunks as numbered context
  - Call LLM and return answer with source citations
  - Support multiple retrieval techniques (naive, HyDE, reranking, multi-query)
"""

import json
import logging

from core import llm_client
from core.models import Citation, RAGResponse
from storage import vector_store
from utils.config import TOP_K, RERANK_CANDIDATES, MULTI_QUERY_VARIANTS

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question using ONLY the provided source documents.

Rules:
- Cite your sources using numbered references like [1], [2], etc.
- Each number corresponds to the source documents listed below.
- If the sources don't contain enough information to answer, say so honestly.
- Do not make up information that isn't in the sources.

Source documents:
{sources}"""


def _build_sources_text(docs: list[str], metas: list[dict]) -> str:
    """Format retrieved chunks as numbered sources for the prompt."""
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        source = meta.get("source_name", "unknown")
        parts.append(f"[{i}] (from {source}):\n{doc}")
    return "\n\n".join(parts)


def _build_citations(docs: list[str], metas: list[dict], distances: list[float]) -> list[Citation]:
    """Convert retrieval results to Citation dataclasses."""
    citations = []
    for doc, meta, dist in zip(docs, metas, distances):
        relevance = max(0.0, 1.0 - dist) if dist is not None else 0.0
        citations.append(Citation(
            source_name=meta.get("source_name", "unknown"),
            chunk_text=doc,
            chunk_index=meta.get("chunk_index", 0),
            relevance_score=round(relevance, 4),
        ))
    return citations


def _naive_retrieve(username: str, notebook_id: str, question: str, top_k: int = TOP_K) -> dict:
    """Naive retrieval: direct vector similarity search."""
    return vector_store.query_collection(username, notebook_id, question, n_results=top_k)


def _hyde_retrieve(username: str, notebook_id: str, question: str, top_k: int = TOP_K) -> dict:
    """HyDE: generate hypothetical answer, embed it, use as query vector."""
    hyde_prompt = (
        "Write a short paragraph that would be an ideal answer to this question. "
        "Do not say 'I don't know'. Just write a plausible, detailed answer.\n\n"
        f"Question: {question}"
    )
    hypothetical = llm_client.complete(
        prompt=hyde_prompt,
        system_prompt="You are a knowledgeable assistant.",
        temperature=0.7,
    )
    return vector_store.query_collection(
        username, notebook_id, hypothetical.text, n_results=top_k
    )


def _reranking_retrieve(username: str, notebook_id: str, question: str, top_k: int = TOP_K) -> dict:
    """Reranking: retrieve top-N candidates, LLM scores relevance, return top-k."""
    initial = vector_store.query_collection(
        username, notebook_id, question, n_results=RERANK_CANDIDATES
    )

    docs = initial.get("documents", [[]])[0]
    metas = initial.get("metadatas", [[]])[0]

    if not docs:
        return initial

    chunks_text = "\n".join(
        f"Chunk {i}: {doc[:300]}" for i, doc in enumerate(docs)
    )
    rerank_prompt = (
        f"Rate each chunk's relevance to the question on a scale of 1-5 "
        f"(5=highly relevant). Return ONLY a JSON array of integers.\n\n"
        f"Question: {question}\n\n{chunks_text}"
    )

    response = llm_client.complete(
        prompt=rerank_prompt,
        system_prompt="You are a relevance scorer. Return only a JSON array of integers.",
        temperature=0.0,
    )

    try:
        scores = json.loads(response.text.strip())
        if not isinstance(scores, list):
            scores = [3] * len(docs)
    except (json.JSONDecodeError, ValueError):
        scores = [3] * len(docs)

    scores = scores[:len(docs)]
    while len(scores) < len(docs):
        scores.append(3)

    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]

    return {
        "documents": [[docs[i] for i in ranked]],
        "metadatas": [[metas[i] for i in ranked]],
        "distances": [[1.0 - (scores[i] / 5.0) for i in ranked]],
    }


def _multi_query_retrieve(username: str, notebook_id: str, question: str, top_k: int = TOP_K) -> dict:
    """Multi-query: generate question variants, retrieve for each, merge via RRF."""
    variant_prompt = (
        f"Generate {MULTI_QUERY_VARIANTS} alternative phrasings of this question. "
        f"Return ONLY a JSON array of strings.\n\nQuestion: {question}"
    )
    response = llm_client.complete(
        prompt=variant_prompt,
        system_prompt="You rephrase questions. Return only a JSON array of strings.",
        temperature=0.7,
    )

    try:
        variants = json.loads(response.text.strip())
        if not isinstance(variants, list):
            variants = []
    except (json.JSONDecodeError, ValueError):
        variants = []

    all_queries = [question] + variants[:MULTI_QUERY_VARIANTS]

    doc_scores: dict[str, float] = {}
    doc_data: dict[str, tuple[str, dict]] = {}

    for q in all_queries:
        results = vector_store.query_collection(username, notebook_id, q, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for rank, (doc, meta) in enumerate(zip(docs, metas)):
            chunk_id = f"{meta.get('source_name', '')}_{meta.get('chunk_index', 0)}"
            doc_scores[chunk_id] = doc_scores.get(chunk_id, 0) + 1.0 / (rank + 60)
            if chunk_id not in doc_data:
                doc_data[chunk_id] = (doc, meta)

    sorted_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)[:top_k]

    return {
        "documents": [[doc_data[cid][0] for cid in sorted_ids]],
        "metadatas": [[doc_data[cid][1] for cid in sorted_ids]],
        "distances": [[1.0 - doc_scores[cid] for cid in sorted_ids]],
    }


_TECHNIQUE_MAP = {
    "naive": _naive_retrieve,
    "hyde": _hyde_retrieve,
    "reranking": _reranking_retrieve,
    "multi_query": _multi_query_retrieve,
}


def query(username: str, notebook_id: str, question: str, technique: str = "naive") -> RAGResponse:
    """Run a RAG query and return the answer with citations."""
    if technique not in _TECHNIQUE_MAP:
        logger.warning("Unknown technique '%s', falling back to naive", technique)
        technique = "naive"

    retrieve_fn = _TECHNIQUE_MAP[technique]
    results = retrieve_fn(username, notebook_id, question)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return RAGResponse(
            answer="No documents have been added to this notebook yet.",
            citations=[],
            technique=technique,
            chunks_considered=0,
        )

    citations = _build_citations(docs, metas, distances)
    sources_text = _build_sources_text(docs, metas)
    system_prompt = RAG_SYSTEM_PROMPT.format(sources=sources_text)

    response = llm_client.complete(
        prompt=question,
        system_prompt=system_prompt,
    )

    return RAGResponse(
        answer=response.text,
        citations=citations,
        technique=technique,
        chunks_considered=len(docs),
    )
