"""Tests for storage/ modules"""

# TODO: Test notebook CRUD (create, list, get, delete)
# TODO: Test chat store (append, read history)
# TODO: Test vector store (add, query, delete)
# TODO: Test artifact store (save, list, get)

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Patch DATA_ROOT before any project module is imported.
# Every test runs against a fresh temporary directory — the real /data volume
# is never touched.
# ---------------------------------------------------------------------------
_TMP_ROOT = tempfile.mkdtemp(prefix="studypod_tests_")
os.environ["DATA_ROOT"] = _TMP_ROOT

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from utils import config as _cfg          # noqa: E402  (must come after env patch)
_cfg.DATA_ROOT = Path(_TMP_ROOT)

import storage.notebook_store as nb_store  # noqa: E402
nb_store.DATA_ROOT = _cfg.DATA_ROOT

import storage.chat_store as chat_store    # noqa: E402
import storage.artifact_store as art_store # noqa: E402

# vector_store requires chromadb; skip gracefully if not installed
try:
    import storage.vector_store as vec_store
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared test user
# ---------------------------------------------------------------------------
USER = "test_user"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def notebook():
    """Create a notebook, yield its metadata, then clean up."""
    import uuid
    meta = nb_store.create_notebook(USER, f"Test NB {uuid.uuid4().hex[:6]}")
    yield meta
    try:
        nb_store.delete_notebook(USER, meta["id"])
    except KeyError:
        pass   # test may have deleted it already


# ===========================================================================
# TODO: Test notebook CRUD (create, list, get, delete)
# ===========================================================================

class TestCreateNotebook:
    def test_returns_metadata_dict(self):
        meta = nb_store.create_notebook(USER, "Create Test")
        assert isinstance(meta, dict)
        assert "id" in meta
        assert "name" in meta
        assert "created_at" in meta
        assert "updated_at" in meta
        nb_store.delete_notebook(USER, meta["id"])

    def test_name_stored_correctly(self):
        meta = nb_store.create_notebook(USER, "Name Check")
        assert meta["name"] == "Name Check"
        nb_store.delete_notebook(USER, meta["id"])

    def test_id_is_uuid4(self):
        import re
        meta = nb_store.create_notebook(USER, "UUID Test")
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(pattern, meta["id"])
        nb_store.delete_notebook(USER, meta["id"])

    def test_duplicate_name_raises_runtime_error(self):
        meta = nb_store.create_notebook(USER, "Dup Name")
        with pytest.raises(RuntimeError):
            nb_store.create_notebook(USER, "Dup Name")
        nb_store.delete_notebook(USER, meta["id"])

    def test_duplicate_name_case_insensitive(self):
        meta = nb_store.create_notebook(USER, "Case NB")
        with pytest.raises(RuntimeError):
            nb_store.create_notebook(USER, "case nb")
        nb_store.delete_notebook(USER, meta["id"])

    def test_directory_tree_created(self, notebook):
        nb_id = notebook["id"]
        assert nb_store.get_raw_dir(USER, nb_id).exists()
        assert nb_store.get_extracted_dir(USER, nb_id).exists()
        assert nb_store.get_chroma_dir(USER, nb_id).exists()
        assert nb_store.get_chat_dir(USER, nb_id).exists()
        for atype in ("reports", "quizzes", "podcasts"):
            assert nb_store.get_artifact_dir(USER, nb_id, atype).exists()

    def test_metadata_json_written(self, notebook):
        meta_file = nb_store.get_notebook_dir(USER, notebook["id"]) / "metadata.json"
        assert meta_file.exists()
        data = json.loads(meta_file.read_text())
        assert data["id"] == notebook["id"]

    def test_index_json_written(self, notebook):
        # index.json must exist after first notebook is created
        user_nb_dir = _cfg.DATA_ROOT / "users" / "test_user" / "notebooks"
        assert (user_nb_dir / "index.json").exists()

    def test_no_orphan_temp_files(self, notebook):
        user_nb_dir = _cfg.DATA_ROOT / "users" / "test_user" / "notebooks"
        orphans = list(user_nb_dir.glob(".tmp_*"))
        assert orphans == [], f"Orphan temp files found: {orphans}"

    def test_blank_name_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            nb_store.create_notebook(USER, "")

    def test_whitespace_only_name_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            nb_store.create_notebook(USER, "   ")


class TestListNotebooks:
    def test_returns_list(self):
        result = nb_store.list_notebooks(USER)
        assert isinstance(result, list)

    def test_created_notebook_appears(self, notebook):
        ids = [nb["id"] for nb in nb_store.list_notebooks(USER)]
        assert notebook["id"] in ids

    def test_sorted_newest_first(self):
        import time
        a = nb_store.create_notebook(USER, "List Sort A")
        time.sleep(0.01)
        b = nb_store.create_notebook(USER, "List Sort B")
        listed = nb_store.list_notebooks(USER)
        ids = [nb["id"] for nb in listed]
        assert ids.index(b["id"]) < ids.index(a["id"])
        nb_store.delete_notebook(USER, a["id"])
        nb_store.delete_notebook(USER, b["id"])

    def test_empty_user_returns_empty_list(self):
        result = nb_store.list_notebooks("nonexistent_user_xyz")
        assert result == []


class TestGetNotebook:
    def test_returns_correct_metadata(self, notebook):
        found = nb_store.get_notebook(USER, notebook["id"])
        assert found["id"] == notebook["id"]
        assert found["name"] == notebook["name"]

    def test_missing_id_raises_key_error(self):
        with pytest.raises(KeyError):
            nb_store.get_notebook(USER, "00000000-0000-0000-0000-000000000000")

    def test_reads_from_metadata_json(self, notebook):
        # Manually corrupt the index to prove get_notebook reads metadata.json
        index_path = _cfg.DATA_ROOT / "users" / "test_user" / "notebooks" / "index.json"
        index = json.loads(index_path.read_text())
        original_name = None
        for entry in index:
            if entry["id"] == notebook["id"]:
                original_name = entry["name"]
                entry["name"] = "CORRUPTED"
                break
        index_path.write_text(json.dumps(index))
        # get_notebook should still return the real name from metadata.json
        found = nb_store.get_notebook(USER, notebook["id"])
        assert found["name"] == original_name
        # Restore index
        for entry in index:
            if entry["id"] == notebook["id"]:
                entry["name"] = original_name
        index_path.write_text(json.dumps(index))


class TestDeleteNotebook:
    def test_removes_from_index(self, notebook):
        nb_id = notebook["id"]
        nb_store.delete_notebook(USER, nb_id)
        ids = [nb["id"] for nb in nb_store.list_notebooks(USER)]
        assert nb_id not in ids

    def test_removes_directory(self, notebook):
        nb_dir = nb_store.get_notebook_dir(USER, notebook["id"])
        assert nb_dir.exists()
        nb_store.delete_notebook(USER, notebook["id"])
        assert not nb_dir.exists()

    def test_get_after_delete_raises(self, notebook):
        nb_store.delete_notebook(USER, notebook["id"])
        with pytest.raises(KeyError):
            nb_store.get_notebook(USER, notebook["id"])

    def test_missing_id_raises_key_error(self):
        with pytest.raises(KeyError):
            nb_store.delete_notebook(USER, "phantom-id-xyz")

    def test_delete_is_idempotent_safe(self, notebook):
        # First delete succeeds, second raises KeyError (not a silent no-op)
        nb_store.delete_notebook(USER, notebook["id"])
        with pytest.raises(KeyError):
            nb_store.delete_notebook(USER, notebook["id"])


# ===========================================================================
# TODO: Test chat store (append, read history)
# ===========================================================================

class TestAppendMessage:
    def test_user_message_appended(self, notebook):
        chat_store.append_message(
            USER, notebook["id"], {"role": "user", "content": "Hello"}
        )
        history = chat_store.get_history(USER, notebook["id"])
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_assistant_message_with_citations(self, notebook):
        msg = {
            "role": "assistant",
            "content": "Here is your answer.",
            "citations": [{"source": "doc.pdf", "chunk": "...", "score": 0.95}],
            "rag_technique": "naive",
            "timing": {"retrieval_ms": 120.0, "generation_ms": 800.0, "total_ms": 920.0},
        }
        chat_store.append_message(USER, notebook["id"], msg)
        history = chat_store.get_history(USER, notebook["id"])
        saved = history[0]
        assert saved["role"] == "assistant"
        assert saved["citations"][0]["source"] == "doc.pdf"
        assert saved["rag_technique"] == "naive"
        assert saved["timing"]["total_ms"] == 920.0

    def test_timestamp_added_automatically(self, notebook):
        chat_store.append_message(
            USER, notebook["id"], {"role": "user", "content": "ts test"}
        )
        history = chat_store.get_history(USER, notebook["id"])
        assert "timestamp" in history[0]
        assert history[0]["timestamp"]  # non-empty

    def test_caller_dict_not_mutated(self, notebook):
        original = {"role": "user", "content": "mutation check"}
        before_keys = set(original.keys())
        chat_store.append_message(USER, notebook["id"], original)
        assert set(original.keys()) == before_keys  # no "timestamp" added to caller's dict

    def test_invalid_role_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            chat_store.append_message(
                USER, notebook["id"], {"role": "system", "content": "oops"}
            )

    def test_missing_role_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            chat_store.append_message(
                USER, notebook["id"], {"content": "no role"}
            )

    def test_missing_content_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            chat_store.append_message(
                USER, notebook["id"], {"role": "user"}
            )

    def test_empty_content_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            chat_store.append_message(
                USER, notebook["id"], {"role": "user", "content": "   "}
            )

    def test_messages_stored_as_jsonl(self, notebook):
        chat_store.append_message(USER, notebook["id"], {"role": "user", "content": "A"})
        chat_store.append_message(USER, notebook["id"], {"role": "assistant", "content": "B"})
        msg_file = nb_store.get_chat_dir(USER, notebook["id"]) / "messages.jsonl"
        lines = [l for l in msg_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        # Each line must be valid JSON
        for line in lines:
            json.loads(line)


class TestGetHistory:
    def test_empty_history_returns_empty_list(self, notebook):
        assert chat_store.get_history(USER, notebook["id"]) == []

    def test_returns_messages_in_order(self, notebook):
        for i in range(4):
            chat_store.append_message(
                USER, notebook["id"], {"role": "user", "content": f"msg {i}"}
            )
        history = chat_store.get_history(USER, notebook["id"])
        assert len(history) == 4
        assert history[0]["content"] == "msg 0"
        assert history[3]["content"] == "msg 3"

    def test_all_fields_preserved(self, notebook):
        msg = {
            "role": "assistant",
            "content": "answer",
            "citations": [{"source": "x.pdf"}],
            "rag_technique": "rerank",
        }
        chat_store.append_message(USER, notebook["id"], msg)
        saved = chat_store.get_history(USER, notebook["id"])[0]
        assert saved["citations"] == [{"source": "x.pdf"}]
        assert saved["rag_technique"] == "rerank"

    def test_survives_corrupt_line(self, notebook):
        # Write one good message, then a corrupt line, then another good message
        chat_store.append_message(USER, notebook["id"], {"role": "user", "content": "before"})
        msg_file = nb_store.get_chat_dir(USER, notebook["id"]) / "messages.jsonl"
        with msg_file.open("a") as f:
            f.write("{{NOT VALID JSON}}\n")
        chat_store.append_message(USER, notebook["id"], {"role": "user", "content": "after"})

        history = chat_store.get_history(USER, notebook["id"])
        contents = [m["content"] for m in history]
        assert "before" in contents
        assert "after" in contents

    def test_get_history_for_llm_strips_internal_fields(self, notebook):
        chat_store.append_message(
            USER, notebook["id"],
            {"role": "assistant", "content": "answer",
             "citations": [{"source": "f.pdf"}], "rag_technique": "naive"}
        )
        ctx = chat_store.get_history_for_llm(USER, notebook["id"])
        assert ctx[0] == {"role": "assistant", "content": "answer"}

    def test_get_history_for_llm_respects_window(self, notebook):
        for i in range(8):
            chat_store.append_message(
                USER, notebook["id"], {"role": "user", "content": f"m{i}"}
            )
        ctx = chat_store.get_history_for_llm(USER, notebook["id"], window=3)
        assert len(ctx) == 3
        assert ctx[-1]["content"] == "m7"

    def test_clear_history(self, notebook):
        chat_store.append_message(USER, notebook["id"], {"role": "user", "content": "x"})
        chat_store.append_message(USER, notebook["id"], {"role": "user", "content": "y"})
        deleted = chat_store.clear_history(USER, notebook["id"])
        assert deleted == 2
        assert chat_store.get_history(USER, notebook["id"]) == []

    def test_clear_empty_returns_zero(self, notebook):
        assert chat_store.clear_history(USER, notebook["id"]) == 0


# ===========================================================================
# TODO: Test vector store (add, query, delete)
# ===========================================================================

@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestVectorStore:
    def test_get_or_create_collection_returns_collection(self, notebook):
        col = vec_store.get_or_create_collection(USER, notebook["id"])
        assert col is not None
        assert hasattr(col, "count")

    def test_get_or_create_collection_idempotent(self, notebook):
        col1 = vec_store.get_or_create_collection(USER, notebook["id"])
        col2 = vec_store.get_or_create_collection(USER, notebook["id"])
        assert col1.name == col2.name

    def test_add_documents_increases_count(self, notebook):
        chunks = ["The sky is blue.", "Grass is green."]
        metas = [{"source": "test.txt"}, {"source": "test.txt"}]
        vec_store.add_documents(USER, notebook["id"], chunks, metas)
        assert vec_store.collection_count(USER, notebook["id"]) == 2

    def test_add_documents_idempotent(self, notebook):
        chunks = ["Only one chunk."]
        metas = [{"source": "idempotent.txt"}]
        vec_store.add_documents(USER, notebook["id"], chunks, metas)
        vec_store.add_documents(USER, notebook["id"], chunks, metas)  # second call
        assert vec_store.collection_count(USER, notebook["id"]) == 1

    def test_add_documents_empty_raises(self, notebook):
        with pytest.raises(ValueError):
            vec_store.add_documents(USER, notebook["id"], [], [])

    def test_add_documents_length_mismatch_raises(self, notebook):
        with pytest.raises(ValueError):
            vec_store.add_documents(
                USER, notebook["id"],
                ["chunk one", "chunk two"],
                [{"source": "only_one_meta.txt"}],
            )

    def test_query_collection_returns_dict(self, notebook):
        vec_store.add_documents(
            USER, notebook["id"],
            ["Python is a programming language."],
            [{"source": "py.txt"}],
        )
        result = vec_store.query_collection(USER, notebook["id"], "programming")
        assert isinstance(result, dict)
        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result

    def test_query_collection_result_shape(self, notebook):
        # ChromaDB returns nested lists — [[...]] not [...]
        vec_store.add_documents(
            USER, notebook["id"],
            ["Machine learning is powerful."],
            [{"source": "ml.txt"}],
        )
        result = vec_store.query_collection(USER, notebook["id"], "machine learning", n_results=1)
        assert isinstance(result["ids"], list)
        assert isinstance(result["ids"][0], list)
        assert len(result["ids"][0]) == 1

    def test_query_collection_empty_collection_returns_empty(self, notebook):
        result = vec_store.query_collection(USER, notebook["id"], "anything")
        assert result["ids"] == [[]]
        assert result["documents"] == [[]]

    def test_query_collection_blank_query_raises(self, notebook):
        with pytest.raises(ValueError):
            vec_store.query_collection(USER, notebook["id"], "   ")

    def test_query_collection_n_results_capped(self, notebook):
        chunks = [f"Chunk number {i}." for i in range(3)]
        metas = [{"source": "cap.txt"}] * 3
        vec_store.add_documents(USER, notebook["id"], chunks, metas)
        result = vec_store.query_collection(USER, notebook["id"], "chunk", n_results=10)
        # Should return at most 3 (the total number of chunks)
        assert len(result["ids"][0]) <= 3

    def test_list_sources(self, notebook):
        vec_store.add_documents(
            USER, notebook["id"],
            ["Alpha text.", "Beta text."],
            [{"source": "alpha.pdf"}, {"source": "beta.pdf"}],
        )
        sources = vec_store.list_sources(USER, notebook["id"])
        assert "alpha.pdf" in sources
        assert "beta.pdf" in sources

    def test_delete_source(self, notebook):
        vec_store.add_documents(
            USER, notebook["id"],
            ["Delete me.", "Keep me."],
            [{"source": "remove.txt"}, {"source": "keep.txt"}],
        )
        deleted = vec_store.delete_source(USER, notebook["id"], "remove.txt")
        assert deleted == 1
        sources = vec_store.list_sources(USER, notebook["id"])
        assert "remove.txt" not in sources
        assert "keep.txt" in sources

    def test_delete_collection(self, notebook):
        vec_store.add_documents(
            USER, notebook["id"],
            ["Some content."],
            [{"source": "col_del.txt"}],
        )
        assert vec_store.collection_count(USER, notebook["id"]) > 0
        vec_store.delete_collection(USER, notebook["id"])
        # After deletion, a fresh collection should start at 0
        assert vec_store.collection_count(USER, notebook["id"]) == 0

    def test_delete_collection_nonexistent_safe(self, notebook):
        # Should not raise even if collection was never created
        vec_store.delete_collection(USER, "00000000-0000-0000-0000-000000000000")


# ===========================================================================
# TODO: Test artifact store (save, list, get)
# ===========================================================================

class TestSaveArtifact:
    def test_returns_path_string(self, notebook):
        path = art_store.save_artifact(
            USER, notebook["id"], "reports", "# Report", "report_1.md"
        )
        assert isinstance(path, str)

    def test_file_exists_on_disk(self, notebook):
        path = art_store.save_artifact(
            USER, notebook["id"], "reports", "# Hello", "report_1.md"
        )
        assert Path(path).exists()

    def test_text_content_written_correctly(self, notebook):
        art_store.save_artifact(
            USER, notebook["id"], "quizzes", "Q1: What is AI?", "quiz_1.md"
        )
        items = art_store.list_artifacts(USER, notebook["id"], "quizzes")
        content = art_store.get_artifact(USER, notebook["id"], "quizzes", items[0]["filename"])
        assert content == "Q1: What is AI?"

    def test_bytes_content_for_podcast_mp3(self, notebook):
        fake_audio = b"\x00\xFF\xFE\x01\x02\x03"
        art_store.save_artifact(
            USER, notebook["id"], "podcasts", fake_audio, "podcast_1.mp3"
        )
        raw = art_store.get_artifact_bytes(USER, notebook["id"], "podcasts", "podcast_1.mp3")
        assert raw == fake_audio

    def test_podcast_transcript_md(self, notebook):
        art_store.save_artifact(
            USER, notebook["id"], "podcasts", "# Transcript text", "podcast_1.md"
        )
        content = art_store.get_artifact(USER, notebook["id"], "podcasts", "podcast_1.md")
        assert "Transcript text" in content

    def test_invalid_type_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            art_store.save_artifact(
                USER, notebook["id"], "flashcards", "content", "flash_1.md"
            )

    def test_wrong_extension_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            art_store.save_artifact(
                USER, notebook["id"], "reports", "content", "report.exe"
            )

    def test_path_traversal_in_filename_is_safe(self, notebook):
        # The crafted filename should be stripped to just the basename
        path = art_store.save_artifact(
            USER, notebook["id"], "reports", "safe", "../../escape.md"
        )
        # Must still land inside the reports/ directory
        assert "reports" in path


class TestListArtifacts:
    def test_empty_returns_empty_list(self, notebook):
        result = art_store.list_artifacts(USER, notebook["id"], "reports")
        assert result == []

    def test_saved_artifact_appears(self, notebook):
        art_store.save_artifact(USER, notebook["id"], "reports", "# R", "report_1.md")
        items = art_store.list_artifacts(USER, notebook["id"], "reports")
        assert len(items) == 1

    def test_item_has_required_keys(self, notebook):
        art_store.save_artifact(USER, notebook["id"], "quizzes", "Q", "quiz_1.md")
        item = art_store.list_artifacts(USER, notebook["id"], "quizzes")[0]
        assert "filename" in item
        assert "path" in item
        assert "type" in item

    def test_filtered_by_type(self, notebook):
        art_store.save_artifact(USER, notebook["id"], "reports", "r", "report_1.md")
        art_store.save_artifact(USER, notebook["id"], "quizzes", "q", "quiz_1.md")
        reports = art_store.list_artifacts(USER, notebook["id"], "reports")
        assert all(i["type"] == "reports" for i in reports)
        assert len(reports) == 1

    def test_no_filter_returns_all_types(self, notebook):
        art_store.save_artifact(USER, notebook["id"], "reports", "r", "report_1.md")
        art_store.save_artifact(USER, notebook["id"], "quizzes", "q", "quiz_1.md")
        all_items = art_store.list_artifacts(USER, notebook["id"])
        types = {i["type"] for i in all_items}
        assert "reports" in types
        assert "quizzes" in types

    def test_invalid_type_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            art_store.list_artifacts(USER, notebook["id"], "videos")


class TestGetArtifact:
    def test_returns_content_string(self, notebook):
        art_store.save_artifact(USER, notebook["id"], "reports", "# Content", "report_1.md")
        content = art_store.get_artifact(USER, notebook["id"], "reports", "report_1.md")
        assert isinstance(content, str)
        assert content == "# Content"

    def test_missing_file_raises_file_not_found(self, notebook):
        with pytest.raises(FileNotFoundError):
            art_store.get_artifact(USER, notebook["id"], "reports", "report_999.md")

    def test_wrong_extension_raises_value_error(self, notebook):
        with pytest.raises(ValueError):
            art_store.get_artifact(USER, notebook["id"], "reports", "report.sh")

    def test_delete_artifact(self, notebook):
        art_store.save_artifact(USER, notebook["id"], "reports", "bye", "report_1.md")
        deleted = art_store.delete_artifact(USER, notebook["id"], "reports", "report_1.md")
        assert deleted is True
        assert art_store.list_artifacts(USER, notebook["id"], "reports") == []

    def test_delete_missing_returns_false(self, notebook):
        result = art_store.delete_artifact(USER, notebook["id"], "reports", "report_999.md")
        assert result is False