"""Tests for core/llm_client.py — mock-based, no API key required."""

from unittest.mock import patch, MagicMock
import pytest

from core.models import LLMResponse, LLMUnavailableError


def _make_mock_completion(text="Hello!", model="llama-3.1-70b-versatile"):
    """Create a mock ChatCompletion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = text
    mock.model = model
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.usage.total_tokens = 15
    return mock


def _make_mock_stream_chunks(texts):
    """Create mock streaming ChatCompletionChunk objects."""
    chunks = []
    for t in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = t
        chunks.append(chunk)
    return iter(chunks)


def _make_rate_limit_error():
    """Create a mock rate limit error."""
    from groq import RateLimitError
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    return RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _make_api_error():
    """Create a mock API error (5xx)."""
    from groq import APIStatusError
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "Internal server error"}}
    return APIStatusError(
        message="Internal server error",
        response=mock_response,
        body={"error": {"message": "Internal server error"}},
    )


# ── T008: Basic completion and streaming ─────────────────────

@patch("core.llm_client._get_client")
def test_complete_returns_llm_response(mock_get_client):
    """complete() returns an LLMResponse with correct fields."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_completion()
    mock_get_client.return_value = mock_client

    from core.llm_client import complete
    response = complete("Say hello")

    assert isinstance(response, LLMResponse)
    assert response.text == "Hello!"
    assert response.model == "llama-3.1-70b-versatile"
    assert response.fallback_used is False
    assert "total_tokens" in response.usage


@patch("core.llm_client._get_client")
def test_streaming_yields_chunks(mock_get_client):
    """complete(stream=True) yields string chunks."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_stream_chunks(
        ["Hello", " world", "!"]
    )
    mock_get_client.return_value = mock_client

    from core.llm_client import complete
    result = list(complete("Say hello", stream=True))

    assert result == ["Hello", " world", "!"]


# ── T009: Retry, fallback, and error handling ────────────────

@patch("core.llm_client.time.sleep")
@patch("core.llm_client._get_client")
def test_retry_on_rate_limit(mock_get_client, mock_sleep):
    """complete() retries on rate limit then succeeds."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_rate_limit_error(),
        _make_mock_completion(),
    ]
    mock_get_client.return_value = mock_client

    from core.llm_client import complete
    response = complete("Say hello")

    assert isinstance(response, LLMResponse)
    assert response.text == "Hello!"
    assert mock_sleep.called


@patch("core.llm_client.time.sleep")
@patch("core.llm_client._get_client")
def test_fallback_on_primary_failure(mock_get_client, mock_sleep):
    """complete() falls back to smaller model when primary exhausts retries."""
    mock_client = MagicMock()
    rate_err = _make_rate_limit_error()
    fallback_response = _make_mock_completion(
        text="Fallback hello", model="llama-3.1-8b-instant"
    )
    # Primary fails 4 times (initial + 3 retries), fallback succeeds
    mock_client.chat.completions.create.side_effect = [
        rate_err, rate_err, rate_err, rate_err,
        fallback_response,
    ]
    mock_get_client.return_value = mock_client

    from core.llm_client import complete
    response = complete("Say hello")

    assert response.text == "Fallback hello"
    assert response.fallback_used is True


@patch("core.llm_client.time.sleep")
@patch("core.llm_client._get_client")
def test_both_models_unavailable_raises_error(mock_get_client, mock_sleep):
    """complete() raises LLMUnavailableError when all models fail."""
    mock_client = MagicMock()
    rate_err = _make_rate_limit_error()
    # Primary fails 4 times, fallback fails 4 times
    mock_client.chat.completions.create.side_effect = [rate_err] * 8
    mock_get_client.return_value = mock_client

    from core.llm_client import complete
    with pytest.raises(LLMUnavailableError):
        complete("Say hello")
