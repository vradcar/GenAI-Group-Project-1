"""
RAG Engine.

Responsibilities:
  - Embed the user's query with the same local model
  - Retrieve top-k (k=5) chunks from ChromaDB by cosine similarity
  - Assemble the prompt: system instructions, retrieved chunks, chat history, question
  - Stream the response from Groq (Llama 3.1 70B)
  - Parse inline citations from the response
  - Append both messages to chat history (JSONL)

RAG techniques to implement:
  1. Naive RAG (baseline)
  2. HyDE (hypothetical document embeddings)
  3. Reranking (cross-encoder)
  4. Multi-Query (LLM-generated query variants)
"""


def query(username: str, notebook_id: str, question: str, technique: str = "naive") -> str:
    """Run a RAG query and return the answer with citations."""
    # TODO
    pass
