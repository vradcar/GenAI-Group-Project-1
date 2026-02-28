"""
Minimal test UI for LLM-RAG functionality.
Paste text directly (bypasses unbuilt ingestion/extractors).
Run: python test_ui.py
"""

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter

from storage.vector_store import add_documents, query_collection, delete_collection
from core.rag import query as rag_query
from core.llm_client import complete
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP

TEST_USER = "testuser"
TEST_NOTEBOOK = "test-notebook"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def ingest_text(text, source_name):
    """Chunk and embed pasted text into the test notebook."""
    if not text.strip():
        return "Please paste some text."
    name = source_name.strip() or "pasted_source.txt"
    chunks = splitter.split_text(text)
    metas = [{"source_name": name, "chunk_index": i} for i in range(len(chunks))]
    add_documents(TEST_USER, TEST_NOTEBOOK, chunks, metas)
    return f"Ingested {len(chunks)} chunks from '{name}'."


def chat(question, technique, history):
    """Run a RAG query and return the answer with citations."""
    if not question.strip():
        return history, ""
    response = rag_query(TEST_USER, TEST_NOTEBOOK, question, technique=technique)
    citations = "\n".join(
        f"  [{i+1}] {c.source_name} (chunk {c.chunk_index}, score {c.relevance_score:.2f})"
        for i, c in enumerate(response.citations)
    )
    answer = response.answer
    if citations:
        answer += f"\n\n--- Sources ({response.technique}) ---\n{citations}"
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def test_llm(prompt):
    """Direct LLM call (no RAG)."""
    if not prompt.strip():
        return "Enter a prompt."
    resp = complete(prompt)
    return f"[{resp.model}, fallback={resp.fallback_used}]\n\n{resp.text}"


def clear_store():
    """Delete all documents from the test notebook."""
    delete_collection(TEST_USER, TEST_NOTEBOOK)
    return "Vector store cleared."


with gr.Blocks(title="RAG Test UI") as app:
    gr.Markdown("## RAG-LLM Test UI\nMinimal interface to test ingestion, retrieval, and chat.")

    with gr.Tab("1. Ingest Text"):
        gr.Markdown("Paste document text below. It will be chunked and embedded.")
        source_name = gr.Textbox(label="Source name", value="document.txt", scale=1)
        text_input = gr.Textbox(label="Paste text here", lines=10)
        ingest_btn = gr.Button("Ingest")
        ingest_status = gr.Textbox(label="Status", interactive=False)
        ingest_btn.click(ingest_text, [text_input, source_name], ingest_status)

    with gr.Tab("2. RAG Chat"):
        technique = gr.Radio(
            ["naive", "hyde", "reranking", "multi_query"],
            value="naive", label="Retrieval technique"
        )
        chatbot = gr.Chatbot(label="Chat")
        question = gr.Textbox(label="Ask a question", placeholder="What is this about?")
        question.submit(chat, [question, technique, chatbot], [chatbot, question])

    with gr.Tab("3. Direct LLM"):
        gr.Markdown("Test the LLM client directly (no RAG).")
        prompt = gr.Textbox(label="Prompt", lines=3)
        llm_btn = gr.Button("Send")
        llm_output = gr.Textbox(label="Response", lines=8, interactive=False)
        llm_btn.click(test_llm, prompt, llm_output)

    with gr.Tab("4. Reset"):
        clear_btn = gr.Button("Clear Vector Store")
        clear_status = gr.Textbox(label="Status", interactive=False)
        clear_btn.click(clear_store, None, clear_status)

if __name__ == "__main__":
    app.launch()
