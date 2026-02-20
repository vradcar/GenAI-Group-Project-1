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


def build_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    # TODO: Build UI layout
    # TODO: Wire callbacks
    pass


if __name__ == "__main__":
    app = build_app()
    app.launch()
