"""
LLM Client — Groq API wrapper.

Responsibilities:
  - Initialize the Groq client (OpenAI-compatible SDK)
  - Send chat completion requests to Llama 3.1 70B
  - Retry with exponential backoff on rate-limit errors
  - Fall back to Llama 3.1 8B if 70B is unavailable
  - Stream responses for real-time chat
"""

import logging
import time
from typing import Generator

from groq import Groq, RateLimitError, APIStatusError

from core.models import LLMResponse, LLMUnavailableError
from utils.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    """Get or create the Groq client singleton."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _call_with_retry(
    messages: list[dict],
    model: str,
    stream: bool,
    temperature: float,
    max_tokens: int,
):
    """Attempt an API call with exponential backoff on transient errors."""
    client = _get_client()
    last_error = None

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES:
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d, model=%s): %s. Retrying in %.1fs",
                    attempt + 1, LLM_MAX_RETRIES + 1, model, e, delay,
                )
                time.sleep(delay)
        except APIStatusError as e:
            # Only retry on server errors (5xx); raise client errors (4xx) immediately
            if e.status_code >= 500:
                last_error = e
                if attempt < LLM_MAX_RETRIES:
                    delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Server error %d (attempt %d/%d, model=%s): %s. Retrying in %.1fs",
                        e.status_code, attempt + 1, LLM_MAX_RETRIES + 1, model, e, delay,
                    )
                    time.sleep(delay)
            else:
                raise

    raise last_error


def complete(
    prompt: str,
    system_prompt: str = "",
    stream: bool = False,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> LLMResponse | Generator[str, None, None]:
    """Send a chat completion request to the LLM.

    Args:
        prompt: User message content.
        system_prompt: System instructions.
        stream: If True, returns a generator yielding text chunks.
        temperature: Sampling temperature (0.0-1.0).
        max_tokens: Maximum tokens in response.

    Returns:
        LLMResponse if stream=False, or a generator of str chunks if stream=True.

    Raises:
        LLMUnavailableError: Both primary and fallback models failed.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    models = [
        (LLM_MODEL, False),
        (LLM_FALLBACK_MODEL, True),
    ]

    for model, is_fallback in models:
        try:
            response = _call_with_retry(messages, model, stream, temperature, max_tokens)

            if stream:
                return _stream_response(response)

            text = response.choices[0].message.content or ""

            # Handle empty response: retry once
            if not text.strip():
                logger.warning("Empty response from %s, retrying once", model)
                response = _call_with_retry(messages, model, False, temperature, max_tokens)
                text = response.choices[0].message.content or ""

            return LLMResponse(
                text=text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                fallback_used=is_fallback,
            )
        except (RateLimitError, APIStatusError) as e:
            logger.warning("Model %s exhausted: %s", model, e)
            if is_fallback:
                raise LLMUnavailableError(
                    f"All LLM models unavailable. Last error: {e}"
                ) from e
            continue

    raise LLMUnavailableError("All LLM models unavailable.")


def _stream_response(response) -> Generator[str, None, None]:
    """Yield text chunks from a streaming response."""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
