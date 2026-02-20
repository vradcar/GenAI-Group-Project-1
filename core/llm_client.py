"""
LLM Client — Groq API wrapper.

Responsibilities:
  - Initialize the Groq client (OpenAI-compatible SDK)
  - Send chat completion requests to Llama 3.1 70B
  - Retry with exponential backoff on rate-limit errors
  - Fall back to Llama 3.1 8B if 70B is unavailable
  - Stream responses for real-time chat
"""

from utils.config import GROQ_API_KEY, LLM_MODEL, LLM_FALLBACK_MODEL


def chat_completion(messages: list[dict], stream: bool = False):
    """Send messages to Groq and return the response (or stream)."""
    # TODO
    pass
