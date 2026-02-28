"""Optional live smoke tests — skipped when GROQ_API_KEY is not set."""

import os
import pytest

requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live smoke test",
)


@requires_api_key
def test_live_complete():
    """Verify LLM client works with real Groq API."""
    from core.llm_client import complete
    from core.models import LLMResponse

    response = complete("Say hello in exactly one sentence.")
    assert isinstance(response, LLMResponse)
    assert len(response.text) > 0
    assert response.model is not None


@requires_api_key
def test_live_streaming():
    """Verify streaming works with real Groq API."""
    from core.llm_client import complete

    chunks = list(complete("Count to 3.", stream=True))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
