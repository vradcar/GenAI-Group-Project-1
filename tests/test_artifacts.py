"""
Unit tests for core/artifacts.py — report, quiz, and podcast generation.

Mocks the LLM client and vector store so tests run without API keys or ChromaDB.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock


from core.models import LLMResponse

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_CHUNKS = [
    "Albert Einstein published the theory of relativity in 1905.",
    "Quantum mechanics describes subatomic particle behavior.",
    "The Standard Model classifies elementary particles.",
]
FAKE_METAS = [
    {"source": "physics.txt", "chunk_index": 0},
    {"source": "physics.txt", "chunk_index": 1},
    {"source": "physics.txt", "chunk_index": 2},
]


def _mock_collection(docs=FAKE_CHUNKS, metas=FAKE_METAS):
    """Return a mock ChromaDB collection with .get() returning docs."""
    coll = MagicMock()
    coll.get.return_value = {
        "documents": docs,
        "metadatas": metas,
    }
    return coll


def _fake_llm(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="fake-model", usage={"total_tokens": 10})


# ---------------------------------------------------------------------------
# _gather_notebook_text
# ---------------------------------------------------------------------------

class TestGatherNotebookText:

    def test_returns_combined_text_with_sources(self):
        from core.artifacts import _gather_notebook_text

        coll = _mock_collection()
        with patch("storage.vector_store.get_or_create_collection", return_value=coll):
            text = _gather_notebook_text("user", "nb1")

        assert "physics.txt" in text
        assert "Einstein" in text
        assert "Quantum" in text

    def test_returns_empty_for_empty_collection(self):
        from core.artifacts import _gather_notebook_text

        coll = _mock_collection(docs=[], metas=[])
        with patch("storage.vector_store.get_or_create_collection", return_value=coll):
            text = _gather_notebook_text("user", "nb1")

        assert text == ""

    def test_handles_exception_gracefully(self):
        from core.artifacts import _gather_notebook_text

        with patch("storage.vector_store.get_or_create_collection", side_effect=Exception("DB error")):
            text = _gather_notebook_text("user", "nb1")

        assert text == ""


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_returns_report_markdown(self, tmp_path):
        from core.artifacts import generate_report

        coll = _mock_collection()
        report_dir = tmp_path / "reports"
        fake_response = _fake_llm("# Physics Report\n\nEinstein was important.")

        with patch("storage.vector_store.get_or_create_collection", return_value=coll), \
             patch("core.artifacts.llm_client.complete", return_value=fake_response), \
             patch("core.artifacts.get_artifact_dir", return_value=report_dir):
            result = generate_report("user", "nb1")

        assert "Physics Report" in result
        assert "Einstein" in result
        # Should have saved a file
        saved = list(report_dir.glob("report_*.md"))
        assert len(saved) == 1

    def test_empty_notebook_returns_no_sources_message(self):
        from core.artifacts import generate_report

        coll = _mock_collection(docs=[], metas=[])
        with patch("storage.vector_store.get_or_create_collection", return_value=coll):
            result = generate_report("user", "nb1")

        assert "No sources" in result

    def test_llm_error_returns_error_message(self):
        from core.artifacts import generate_report

        coll = _mock_collection()
        with patch("storage.vector_store.get_or_create_collection", return_value=coll), \
             patch("core.artifacts.llm_client.complete", side_effect=Exception("API down")), \
             patch("core.artifacts.get_artifact_dir", return_value=Path("/tmp/art")):
            result = generate_report("user", "nb1")

        assert "Error" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# generate_quiz
# ---------------------------------------------------------------------------

class TestGenerateQuiz:

    def test_returns_quiz_markdown(self, tmp_path):
        from core.artifacts import generate_quiz

        coll = _mock_collection()
        quiz_dir = tmp_path / "quizzes"
        fake_response = _fake_llm(
            "## Quiz Questions\n\n1. What year did Einstein publish relativity?\n\n"
            "## Answer Key\n\n1. 1905"
        )

        with patch("storage.vector_store.get_or_create_collection", return_value=coll), \
             patch("core.artifacts.llm_client.complete", return_value=fake_response), \
             patch("core.artifacts.get_artifact_dir", return_value=quiz_dir):
            result = generate_quiz("user", "nb1")

        assert "Quiz" in result
        assert "1905" in result
        saved = list(quiz_dir.glob("quiz_*.md"))
        assert len(saved) == 1

    def test_empty_notebook_returns_no_sources(self):
        from core.artifacts import generate_quiz

        coll = _mock_collection(docs=[], metas=[])
        with patch("storage.vector_store.get_or_create_collection", return_value=coll):
            result = generate_quiz("user", "nb1")

        assert "No sources" in result

    def test_custom_num_questions(self, tmp_path):
        from core.artifacts import generate_quiz

        coll = _mock_collection()
        quiz_dir = tmp_path / "quizzes"
        captured_prompts = []

        def capture_complete(prompt, **kwargs):
            captured_prompts.append(prompt)
            return _fake_llm("## Quiz\n\n1. Q1\n\n## Answer Key\n\n1. A1")

        with patch("storage.vector_store.get_or_create_collection", return_value=coll), \
             patch("core.artifacts.llm_client.complete", side_effect=capture_complete), \
             patch("core.artifacts.get_artifact_dir", return_value=quiz_dir):
            generate_quiz("user", "nb1", num_questions=5)

        # The prompt should mention 5 questions
        assert "5" in captured_prompts[0]


# ---------------------------------------------------------------------------
# generate_podcast
# ---------------------------------------------------------------------------

class TestGeneratePodcast:

    def test_returns_transcript_and_audio_paths(self, tmp_path):
        from core.artifacts import generate_podcast

        coll = _mock_collection()
        podcast_dir = tmp_path / "podcasts"
        fake_response = _fake_llm(
            "[Alex]: Welcome to the show!\n[Sam]: Today we discuss physics."
        )

        async def fake_tts_save(path):
            Path(path).write_bytes(b"fake mp3 data")

        mock_communicate = MagicMock()
        mock_communicate.return_value.save = fake_tts_save

        with patch("storage.vector_store.get_or_create_collection", return_value=coll), \
             patch("core.artifacts.llm_client.complete", return_value=fake_response), \
             patch("core.artifacts.get_artifact_dir", return_value=podcast_dir), \
             patch("edge_tts.Communicate", mock_communicate):
            transcript_path, audio_path = generate_podcast("user", "nb1")

        assert transcript_path is not None
        assert Path(transcript_path).exists()
        assert "transcript" in Path(transcript_path).name

    def test_empty_notebook_returns_none_tuple(self):
        from core.artifacts import generate_podcast

        coll = _mock_collection(docs=[], metas=[])
        with patch("storage.vector_store.get_or_create_collection", return_value=coll):
            result = generate_podcast("user", "nb1")

        assert result == (None, None)

    def test_tts_failure_still_returns_transcript(self, tmp_path):
        """If TTS fails, transcript should still be saved."""
        from core.artifacts import generate_podcast

        coll = _mock_collection()
        podcast_dir = tmp_path / "podcasts"
        fake_response = _fake_llm("[Alex]: Hello!\n[Sam]: Goodbye!")

        with patch("storage.vector_store.get_or_create_collection", return_value=coll), \
             patch("core.artifacts.llm_client.complete", return_value=fake_response), \
             patch("core.artifacts.get_artifact_dir", return_value=podcast_dir), \
             patch("edge_tts.Communicate", side_effect=Exception("TTS broken")):
            transcript_path, audio_path = generate_podcast("user", "nb1")

        assert transcript_path is not None
        assert Path(transcript_path).exists()
        assert audio_path is None


# ---------------------------------------------------------------------------
# _parse_transcript helper
# ---------------------------------------------------------------------------

class TestParseTranscript:
    """Unit tests for the two-speaker transcript parser."""

    def test_basic_two_speakers(self):
        from core.artifacts import _parse_transcript

        text = "[Alex]: Hello there!\n[Sam]: Hey, welcome!"
        segs = _parse_transcript(text)
        assert segs == [("A", "Hello there!"), ("B", "Hey, welcome!")]

    def test_merges_adjacent_same_speaker(self):
        from core.artifacts import _parse_transcript

        text = "[Alex]: Line one.\n[Alex]: Line two.\n[Sam]: Response."
        segs = _parse_transcript(text)
        assert len(segs) == 2
        assert segs[0] == ("A", "Line one. Line two.")
        assert segs[1] == ("B", "Response.")

    def test_continuation_lines_appended(self):
        from core.artifacts import _parse_transcript

        text = "[Sam]: Start here.\nMore of Sam talking.\n[Alex]: Ok!"
        segs = _parse_transcript(text)
        assert len(segs) == 2
        assert "More of Sam talking" in segs[0][1]
        assert segs[1] == ("A", "Ok!")

    def test_case_insensitive(self):
        from core.artifacts import _parse_transcript

        text = "[ALEX]: Yell!\n[sam]: Whisper."
        segs = _parse_transcript(text)
        assert segs == [("A", "Yell!"), ("B", "Whisper.")]

    def test_bold_markdown_labels(self):
        from core.artifacts import _parse_transcript

        text = "**Alex:** Bold label.\n**Sam:** Also bold."
        segs = _parse_transcript(text)
        assert len(segs) == 2
        assert segs[0][0] == "A"
        assert segs[1][0] == "B"

    def test_empty_transcript(self):
        from core.artifacts import _parse_transcript

        assert _parse_transcript("") == []
        assert _parse_transcript("   \n\n  ") == []

    def test_no_speaker_labels(self):
        from core.artifacts import _parse_transcript

        text = "Just some random text with no labels."
        assert _parse_transcript(text) == []