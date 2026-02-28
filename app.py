"""
StudyPod — Main entry point.

Responsibilities:
  - Build the Gradio UI layout (login, notebook sidebar, chat panel, artifact panel)
  - Wire up callbacks to core/ and storage/ modules
  - Manage OAuth session state via gr.OAuthProfile
  - Launch the app

Run:
  python app.py
"""

import logging
import time

import gradio as gr

from core import ingestion, rag
from core.artifacts import generate_report, generate_quiz, generate_podcast
from core.models import LLMUnavailableError
from storage import notebook_store, chat_store, artifact_store, vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_username(profile: gr.OAuthProfile | None) -> str:
    """Extract username from OAuth profile, or return empty string."""
    if profile is None:
        return ""
    return profile.username


def _notebook_choices(username: str) -> list[str]:
    """Return list of 'name (id)' strings for the dropdown."""
    if not username:
        return []
    notebooks = notebook_store.list_notebooks(username)
    return [f"{nb['name']}  ({nb['id']})" for nb in notebooks]


def _parse_notebook_id(selection: str | None) -> str | None:
    """Extract notebook UUID from dropdown value like 'My Notes  (uuid-here)'."""
    if not selection:
        return None
    # ID is between the last '(' and ')'
    try:
        return selection.rsplit("(", 1)[1].rstrip(")")
    except (IndexError, AttributeError):
        return None


def _sources_markdown(username: str, notebook_id: str) -> str:
    """Build a Markdown list of ingested sources for display."""
    sources = vector_store.list_sources(username, notebook_id)
    if not sources:
        return "*No sources ingested yet.*"
    lines = [f"- {s}" for s in sources]
    return "**Ingested sources:**\n" + "\n".join(lines)


def _citations_markdown(citations) -> str:
    """Format RAG citations as Markdown."""
    if not citations:
        return ""
    lines = ["**Sources cited:**"]
    seen = set()
    for c in citations:
        label = c.source_name
        if label not in seen:
            score_pct = f"{c.relevance_score * 100:.0f}%"
            lines.append(f"- **{label}** (relevance: {score_pct})")
            seen.add(label)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="StudyPod — NotebookLM Clone",
        theme=gr.themes.Soft(),
        css="""
            .source-list { max-height: 200px; overflow-y: auto; }
            footer { display: none !important; }
        """,
    ) as demo:

        # ── Hidden state ──────────────────────────────────────
        current_user = gr.State("")
        current_notebook_id = gr.State("")

        # ── Header ────────────────────────────────────────────
        gr.Markdown("# 📚 StudyPod — NotebookLM Clone")

        # ── Auth ──────────────────────────────────────────────
        gr.LoginButton()
        status_md = gr.Markdown("*Please log in with Hugging Face to start.*")

        # ── Main layout ──────────────────────────────────────
        with gr.Row():

            # ============  LEFT SIDEBAR  ============
            with gr.Column(scale=1, min_width=280):

                # -- Notebook management --
                gr.Markdown("### 📓 Notebooks")
                notebook_dd = gr.Dropdown(
                    label="Select Notebook",
                    choices=[],
                    interactive=True,
                )
                with gr.Row():
                    notebook_name_tb = gr.Textbox(
                        label="New Notebook Name",
                        placeholder="e.g. CS5010 Notes",
                        scale=3,
                    )
                    create_nb_btn = gr.Button("Create", scale=1, variant="primary")
                with gr.Row():
                    rename_nb_btn = gr.Button("Rename", size="sm")
                    delete_nb_btn = gr.Button("Delete", size="sm", variant="stop")

                gr.Markdown("---")

                # -- Source management --
                gr.Markdown("### 📁 Sources")
                file_upload = gr.File(
                    label="Upload files (PDF, PPTX, TXT)",
                    file_count="multiple",
                    file_types=[".pdf", ".pptx", ".txt"],
                )
                url_input = gr.Textbox(
                    label="Or paste a URL",
                    placeholder="https://example.com/article",
                )
                ingest_url_btn = gr.Button("Ingest URL", size="sm")
                source_list_md = gr.Markdown("*No sources yet.*", elem_classes="source-list")

                gr.Markdown("---")

                # -- Artifacts sidebar --
                gr.Markdown("### 📦 Artifacts")
                artifact_dd = gr.Dropdown(label="Saved Artifacts", choices=[], interactive=True)
                refresh_artifacts_btn = gr.Button("Refresh list", size="sm")

            # ============  RIGHT MAIN PANEL  ============
            with gr.Column(scale=3):

                with gr.Tabs():

                    # ── Chat tab ──────────────────────
                    with gr.TabItem("💬 Chat"):
                        chatbot = gr.Chatbot(label="Chat", height=420, type="messages")
                        with gr.Row():
                            rag_technique = gr.Dropdown(
                                label="RAG technique",
                                choices=["naive", "hyde", "reranking", "multi_query"],
                                value="naive",
                                scale=1,
                            )
                            user_input = gr.Textbox(
                                label="Ask a question",
                                placeholder="What are the key concepts?",
                                scale=4,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        citation_md = gr.Markdown()

                    # ── Artifacts tab ─────────────────
                    with gr.TabItem("📝 Artifacts"):
                        gr.Markdown("Generate study materials from your notebook sources.")
                        with gr.Row():
                            report_btn = gr.Button("📄 Generate Report", variant="primary")
                            quiz_btn = gr.Button("❓ Generate Quiz", variant="primary")
                            podcast_btn = gr.Button("🎙️ Generate Podcast", variant="primary")
                        artifact_viewer = gr.Markdown(label="Artifact Preview")
                        audio_player = gr.Audio(label="Podcast Audio", visible=False)

        # ==================================================================
        # CALLBACKS
        # ==================================================================

        # ── Auth: on login / page load ────────────────────────
        def on_login(profile: gr.OAuthProfile | None):
            username = _get_username(profile)
            if not username:
                return (
                    "",          # current_user
                    "",          # current_notebook_id
                    gr.update(choices=[], value=None),  # notebook_dd
                    "*Please log in with Hugging Face to start.*",
                    gr.update(choices=[], value=None),  # artifact_dd
                    [],          # chatbot
                    "",          # citation_md
                    "*No sources yet.*",
                    gr.update(visible=False),  # audio_player
                    "",          # artifact_viewer
                )
            choices = _notebook_choices(username)
            return (
                username,
                "",
                gr.update(choices=choices, value=None),
                f"✅ Logged in as **{username}**",
                gr.update(choices=[], value=None),
                [],
                "",
                "*Select or create a notebook.*",
                gr.update(visible=False),
                "",
            )

        demo.load(
            on_login,
            inputs=None,
            outputs=[
                current_user, current_notebook_id, notebook_dd,
                status_md, artifact_dd, chatbot, citation_md,
                source_list_md, audio_player, artifact_viewer,
            ],
        )

        # ── Create notebook ───────────────────────────────────
        def on_create_notebook(username, name):
            if not username:
                gr.Warning("Please log in first.")
                return gr.update(), gr.update(), "", ""
            if not name or not name.strip():
                gr.Warning("Please enter a notebook name.")
                return gr.update(), gr.update(), "", ""
            try:
                meta = notebook_store.create_notebook(username, name.strip())
                choices = _notebook_choices(username)
                # Auto-select the new notebook
                new_val = f"{meta['name']}  ({meta['id']})"
                return (
                    gr.update(choices=choices, value=new_val),
                    meta["id"],
                    "",   # clear name textbox
                    "*No sources yet.*",
                )
            except (ValueError, RuntimeError) as e:
                gr.Warning(str(e))
                return gr.update(), gr.update(), name, gr.update()

        create_nb_btn.click(
            on_create_notebook,
            inputs=[current_user, notebook_name_tb],
            outputs=[notebook_dd, current_notebook_id, notebook_name_tb, source_list_md],
        )

        # ── Select notebook ───────────────────────────────────
        def on_select_notebook(username, selection):
            nb_id = _parse_notebook_id(selection)
            if not username or not nb_id:
                return "", [], "", "*No sources yet.*", gr.update(choices=[], value=None), "", gr.update(visible=False)
            # Load chat history
            history = chat_store.get_history(username, nb_id)
            messages = []
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            # Load sources
            sources_md = _sources_markdown(username, nb_id)
            # Load artifacts list
            arts = artifact_store.list_artifacts(username, nb_id)
            art_choices = [f"{a['type']}/{a['filename']}" for a in arts]
            return (
                nb_id,
                messages,
                "",          # clear citations
                sources_md,
                gr.update(choices=art_choices, value=None),
                "",          # clear artifact viewer
                gr.update(visible=False),  # hide audio
            )

        notebook_dd.change(
            on_select_notebook,
            inputs=[current_user, notebook_dd],
            outputs=[
                current_notebook_id, chatbot, citation_md,
                source_list_md, artifact_dd, artifact_viewer, audio_player,
            ],
        )

        # ── Delete notebook ───────────────────────────────────
        def on_delete_notebook(username, selection):
            nb_id = _parse_notebook_id(selection)
            if not username or not nb_id:
                gr.Warning("No notebook selected.")
                return gr.update(), "", [], "", "*No sources yet.*"
            try:
                notebook_store.delete_notebook(username, nb_id)
                vector_store.delete_collection(username, nb_id)
            except KeyError:
                pass
            choices = _notebook_choices(username)
            return (
                gr.update(choices=choices, value=None),
                "",
                [],
                "",
                "*Select or create a notebook.*",
            )

        delete_nb_btn.click(
            on_delete_notebook,
            inputs=[current_user, notebook_dd],
            outputs=[notebook_dd, current_notebook_id, chatbot, citation_md, source_list_md],
        )

        # ── Rename notebook ───────────────────────────────────
        def on_rename_notebook(username, selection, new_name):
            nb_id = _parse_notebook_id(selection)
            if not username or not nb_id:
                gr.Warning("No notebook selected.")
                return gr.update(), ""
            if not new_name or not new_name.strip():
                gr.Warning("Enter a new name in the name field.")
                return gr.update(), new_name
            try:
                notebook_store.update_notebook_name(username, nb_id, new_name.strip())
                choices = _notebook_choices(username)
                meta = notebook_store.get_notebook(username, nb_id)
                new_val = f"{meta['name']}  ({meta['id']})"
                return gr.update(choices=choices, value=new_val), ""
            except (ValueError, RuntimeError, KeyError) as e:
                gr.Warning(str(e))
                return gr.update(), new_name

        rename_nb_btn.click(
            on_rename_notebook,
            inputs=[current_user, notebook_dd, notebook_name_tb],
            outputs=[notebook_dd, notebook_name_tb],
        )

        # ── File upload (ingestion) ───────────────────────────
        def on_file_upload(username, nb_id, files):
            if not username:
                gr.Warning("Please log in first.")
                return gr.update()
            if not nb_id:
                gr.Warning("Please select or create a notebook first.")
                return gr.update()
            if not files:
                return gr.update()

            results = []
            for f in files:
                try:
                    file_path = f.name if hasattr(f, "name") else str(f)
                    res = ingestion.ingest_file(username, nb_id, file_path)
                    results.append(f"✅ {res['source']} — {res['chunks']} chunks")
                except (ValueError, OSError) as e:
                    results.append(f"❌ Error: {e}")

            gr.Info("\n".join(results))
            return _sources_markdown(username, nb_id)

        file_upload.change(
            on_file_upload,
            inputs=[current_user, current_notebook_id, file_upload],
            outputs=[source_list_md],
        )

        # ── URL ingestion ─────────────────────────────────────
        def on_ingest_url(username, nb_id, url):
            if not username:
                gr.Warning("Please log in first.")
                return gr.update(), url
            if not nb_id:
                gr.Warning("Please select or create a notebook first.")
                return gr.update(), url
            if not url or not url.strip():
                gr.Warning("Please enter a URL.")
                return gr.update(), url
            try:
                res = ingestion.ingest_url(username, nb_id, url.strip())
                gr.Info(f"✅ {res['source']} — {res['chunks']} chunks")
                return _sources_markdown(username, nb_id), ""
            except (ValueError, OSError) as e:
                gr.Warning(f"❌ {e}")
                return gr.update(), url

        ingest_url_btn.click(
            on_ingest_url,
            inputs=[current_user, current_notebook_id, url_input],
            outputs=[source_list_md, url_input],
        )

        # ── Chat (RAG) ───────────────────────────────────────
        def on_chat(username, nb_id, message, history, technique):
            if not username:
                gr.Warning("Please log in first.")
                return history or [], "", gr.update()
            if not nb_id:
                gr.Warning("Please select or create a notebook first.")
                return history or [], "", gr.update()
            if not message or not message.strip():
                return history or [], "", gr.update()

            history = history or []

            # Append user message to display and store
            history.append({"role": "user", "content": message.strip()})
            chat_store.append_message(username, nb_id, {
                "role": "user", "content": message.strip(),
            })

            try:
                t0 = time.time()
                response = rag.query(username, nb_id, message.strip(), technique=technique)
                elapsed_ms = (time.time() - t0) * 1000

                answer = response.answer
                citations_md = _citations_markdown(response.citations)
                citations_md += f"\n\n*Technique: {response.technique} | "
                citations_md += f"Chunks: {response.chunks_considered} | "
                citations_md += f"Time: {elapsed_ms:.0f}ms*"

                # Store assistant message
                chat_store.append_message(username, nb_id, {
                    "role": "assistant",
                    "content": answer,
                    "rag_technique": response.technique,
                    "citations": [
                        {"source": c.source_name, "chunk_index": c.chunk_index,
                         "score": c.relevance_score}
                        for c in response.citations
                    ],
                    "timing": {"total_ms": round(elapsed_ms, 1)},
                })

                history.append({"role": "assistant", "content": answer})
                return history, "", citations_md

            except LLMUnavailableError:
                err = "⚠️ The LLM is currently unavailable. Please try again later."
                history.append({"role": "assistant", "content": err})
                return history, "", ""
            except Exception as e:
                logger.exception("Chat error")
                err = f"⚠️ Error: {e}"
                history.append({"role": "assistant", "content": err})
                return history, "", ""

        send_btn.click(
            on_chat,
            inputs=[current_user, current_notebook_id, user_input, chatbot, rag_technique],
            outputs=[chatbot, user_input, citation_md],
        )
        user_input.submit(
            on_chat,
            inputs=[current_user, current_notebook_id, user_input, chatbot, rag_technique],
            outputs=[chatbot, user_input, citation_md],
        )

        # ── Artifact generation ───────────────────────────────
        def on_generate_report(username, nb_id):
            if not username or not nb_id:
                gr.Warning("Select a notebook first.")
                return "", gr.update(visible=False)
            try:
                result = generate_report(username, nb_id)
                if result is None:
                    return "⚠️ Report generation is not yet implemented.", gr.update(visible=False)
                return result, gr.update(visible=False)
            except Exception as e:
                logger.exception("Report generation error")
                return f"⚠️ Error generating report: {e}", gr.update(visible=False)

        def on_generate_quiz(username, nb_id):
            if not username or not nb_id:
                gr.Warning("Select a notebook first.")
                return "", gr.update(visible=False)
            try:
                result = generate_quiz(username, nb_id)
                if result is None:
                    return "⚠️ Quiz generation is not yet implemented.", gr.update(visible=False)
                return result, gr.update(visible=False)
            except Exception as e:
                logger.exception("Quiz generation error")
                return f"⚠️ Error generating quiz: {e}", gr.update(visible=False)

        def on_generate_podcast(username, nb_id):
            if not username or not nb_id:
                gr.Warning("Select a notebook first.")
                return "", gr.update(visible=True)
            try:
                result = generate_podcast(username, nb_id)
                if result is None:
                    return "⚠️ Podcast generation is not yet implemented.", gr.update(visible=False, value=None)
                transcript_path, audio_path = result
                # Read the transcript markdown
                try:
                    from pathlib import Path
                    transcript = Path(transcript_path).read_text(encoding="utf-8") if transcript_path else ""
                except Exception:
                    transcript = "Podcast generated."
                return transcript, gr.update(visible=True, value=audio_path)
            except Exception as e:
                logger.exception("Podcast generation error")
                return f"⚠️ Error generating podcast: {e}", gr.update(visible=False, value=None)

        report_btn.click(
            on_generate_report,
            inputs=[current_user, current_notebook_id],
            outputs=[artifact_viewer, audio_player],
        )
        quiz_btn.click(
            on_generate_quiz,
            inputs=[current_user, current_notebook_id],
            outputs=[artifact_viewer, audio_player],
        )
        podcast_btn.click(
            on_generate_podcast,
            inputs=[current_user, current_notebook_id],
            outputs=[artifact_viewer, audio_player],
        )

        # ── Artifact list / view ──────────────────────────────
        def on_refresh_artifacts(username, nb_id):
            if not username or not nb_id:
                return gr.update(choices=[], value=None)
            arts = artifact_store.list_artifacts(username, nb_id)
            choices = [f"{a['type']}/{a['filename']}" for a in arts]
            return gr.update(choices=choices, value=None)

        refresh_artifacts_btn.click(
            on_refresh_artifacts,
            inputs=[current_user, current_notebook_id],
            outputs=[artifact_dd],
        )

        def on_select_artifact(username, nb_id, selection):
            if not username or not nb_id or not selection:
                return "", gr.update(visible=False, value=None)
            try:
                art_type, filename = selection.split("/", 1)
                content = artifact_store.get_artifact(username, nb_id, art_type, filename)
                is_audio = filename.endswith(".mp3")
                if is_audio:
                    # We need the file path, not bytes, for gr.Audio
                    from storage.notebook_store import get_artifact_dir
                    fpath = get_artifact_dir(username, nb_id, art_type) / filename
                    return "", gr.update(visible=True, value=str(fpath))
                return content, gr.update(visible=False, value=None)
            except (FileNotFoundError, ValueError) as e:
                return f"⚠️ {e}", gr.update(visible=False, value=None)

        artifact_dd.change(
            on_select_artifact,
            inputs=[current_user, current_notebook_id, artifact_dd],
            outputs=[artifact_viewer, audio_player],
        )

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch()