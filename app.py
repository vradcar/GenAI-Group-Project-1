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

import gradio as gr

user_data = {}  # {username: [notebook1, notebook2,...]}

def get_user_notebooks(username):
    return user_data.get(username, ["Demo Notebook"])

def create_notebook(username, name):
    notebooks = user_data.get(username, ["Demo Notebook"])
    if name and name not in notebooks:
        notebooks.append(name)
    user_data[username] = notebooks
    return notebooks

def delete_notebook(username, name):
    notebooks = user_data.get(username, ["Demo Notebook"])
    if name in notebooks:
        notebooks.remove(name)
    user_data[username] = notebooks
    return notebooks

def chat(username, notebook_id, message, history):
    history = history or []
    reply = f"[Test] {username} asked: {message} in {notebook_id}"
    history.append((message, reply))
    return history, "- Source1.pdf\n- Lecture2.txt"

def generate_report(username, notebook_id):
    return f"# Report for {notebook_id}\nGenerated for {username}"

def generate_quiz(username, notebook_id):
    return f"# Quiz for {notebook_id}\nQ1: What is X?\nA1: X is ..."

def generate_podcast(username, notebook_id):
    transcript = f"# Podcast Transcript for {notebook_id}\nSpeaker 1: Hello!"
    audio_path = None  # Test, replace with TTS later
    return transcript, audio_path

def list_artifacts(username, notebook_id):
    return ["report_1.md", "quiz_1.md", "podcast_1.md"]

def load_artifact(username, notebook_id, artifact_name):
    return f"# {artifact_name} contents for {username}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="StudyPod — NotebookLM Clone") as demo:

        gr.Markdown("# 📚 StudyPod — NotebookLM Clone")

        # Test OAuth login simulation
        username_label = gr.Label("Not logged in")
        login_btn = gr.Button("Login (Test)")
        login_btn.click(lambda: "test_user", inputs=None, outputs=username_label)

        with gr.Row():
            # LEFT PANEL: Notebook Sidebar
            with gr.Column(scale=1):
                notebook_dropdown = gr.Dropdown(label="Select Notebook", choices=[])
                notebook_name = gr.Textbox(label="New Notebook Name")
                create_btn = gr.Button("Create Notebook")
                delete_btn = gr.Button("Delete Notebook")

                gr.Markdown("## Artifacts")
                artifact_list = gr.Dropdown(label="Saved Artifacts")
                refresh_artifacts_btn = gr.Button("Refresh Artifacts")

            # RIGHT PANEL: Chat + Artifacts
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat")
                user_input = gr.Textbox(label="Ask a question")
                send_btn = gr.Button("Send")
                citation_box = gr.Markdown(label="Citations")

                gr.Markdown("## Generate Artifacts")
                report_btn = gr.Button("Generate Report")
                quiz_btn = gr.Button("Generate Quiz")
                podcast_btn = gr.Button("Generate Podcast")

                artifact_viewer = gr.Markdown()
                audio_player = gr.Audio()

        # ------------------------------
        # Callbacks
        # ------------------------------
        # Load notebooks after "login"
        def load_user_notebooks(username):
            return get_user_notebooks(username)
        login_btn.click(load_user_notebooks, inputs=username_label, outputs=notebook_dropdown)

        create_btn.click(create_notebook, inputs=[username_label, notebook_name], outputs=notebook_dropdown)
        delete_btn.click(delete_notebook, inputs=[username_label, notebook_dropdown], outputs=notebook_dropdown)

        send_btn.click(chat, inputs=[username_label, notebook_dropdown, user_input, chatbot], outputs=[chatbot, citation_box])

        report_btn.click(generate_report, inputs=[username_label, notebook_dropdown], outputs=artifact_viewer)
        quiz_btn.click(generate_quiz, inputs=[username_label, notebook_dropdown], outputs=artifact_viewer)
        podcast_btn.click(generate_podcast, inputs=[username_label, notebook_dropdown], outputs=[artifact_viewer, audio_player])

        refresh_artifacts_btn.click(list_artifacts, inputs=[username_label, notebook_dropdown], outputs=artifact_list)
        artifact_list.change(load_artifact, inputs=[username_label, notebook_dropdown, artifact_list], outputs=artifact_viewer)

    return demo

# ------------------------------
# Launch the app
# ------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch()