"""
Shared data models for LLM and RAG modules.
"""

from dataclasses import dataclass, field


class LLMUnavailableError(Exception):
    """Raised when both primary and fallback LLM models are unavailable."""


@dataclass
class LLMResponse:
    """Structured output from an LLM call."""
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    fallback_used: bool = False


@dataclass
class Citation:
    """A reference to a source passage used in a RAG response."""
    source_name: str
    chunk_text: str
    chunk_index: int
    relevance_score: float


@dataclass
class RAGResponse:
    """Complete response from a RAG query."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    technique: str = "naive"
    chunks_considered: int = 0


@dataclass
class RetrievalResult:
    """Output from a vector store query."""
    chunks: list[dict] = field(default_factory=list)
    query_embedding: list[float] = field(default_factory=list)
