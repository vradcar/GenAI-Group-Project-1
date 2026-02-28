"""Tests for utils/security.py"""

# TODO: Test sanitize_filename
# TODO: Test sanitize_notebook_name
# TODO: Test sanitize_username
# TODO: Test validate_path (happy path + path traversal rejection)
# TODO: Test generate_notebook_id (format, uniqueness)

import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security import (
    generate_notebook_id,
    safe_path,
    sanitize_filename,
    sanitize_notebook_name,
    sanitize_username,
    validate_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# ===========================================================================
# TODO: Test sanitize_filename
# ===========================================================================

class TestSanitizeFilename:

    # ── happy path ──────────────────────────────────────────────────────────

    def test_normal_filename_unchanged(self):
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_normal_filename_with_underscores(self):
        assert sanitize_filename("my_file_v2.txt") == "my_file_v2.txt"

    def test_filename_with_numbers(self):
        assert sanitize_filename("chapter01.md") == "chapter01.md"

    def test_returns_string(self):
        assert isinstance(sanitize_filename("file.txt"), str)

    # ── path traversal ───────────────────────────────────────────────────────

    def test_strips_unix_traversal_prefix(self):
        # ../../etc/passwd → only the basename survives
        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_strips_single_slash_prefix(self):
        assert sanitize_filename("/etc/passwd") == "passwd"

    def test_strips_windows_traversal(self):
        result = sanitize_filename("..\\Windows\\System32\\cmd.exe")
        assert ".." not in result
        assert "\\" not in result

    def test_no_directory_separators_in_result(self):
        result = sanitize_filename("a/b/c/file.txt")
        assert "/" not in result

    # ── null bytes and control characters ────────────────────────────────────

    def test_null_byte_removed(self):
        assert "\x00" not in sanitize_filename("fi\x00le.txt")

    def test_control_chars_removed(self):
        result = sanitize_filename("fi\x1ble.txt")
        assert "\x1b" not in result

    def test_all_control_chars_removed(self):
        # Build a name with every control character 0x00-0x1f
        dirty = "".join(chr(i) for i in range(32)) + "file.txt"
        result = sanitize_filename(dirty)
        for i in range(32):
            assert chr(i) not in result

    # ── OS-unsafe characters replaced ────────────────────────────────────────

    def test_angle_brackets_replaced(self):
        result = sanitize_filename("file<n>ame.txt")
        assert "<" not in result
        assert ">" not in result

    def test_colon_replaced(self):
        assert ":" not in sanitize_filename("C:file.txt")

    def test_pipe_replaced(self):
        assert "|" not in sanitize_filename("file|name.txt")

    def test_question_mark_replaced(self):
        assert "?" not in sanitize_filename("file?.txt")

    def test_asterisk_replaced(self):
        assert "*" not in sanitize_filename("fi*le.txt")

    def test_quote_replaced(self):
        assert '"' not in sanitize_filename('say"hello".txt')

    def test_backslash_replaced(self):
        # Backslash is unsafe — replaced, not deleted
        result = sanitize_filename("foo\\bar.txt")
        assert "\\" not in result

    # ── dot handling ─────────────────────────────────────────────────────────

    def test_consecutive_dots_collapsed(self):
        result = sanitize_filename("file...exe")
        assert ".." not in result

    def test_leading_dot_stripped(self):
        result = sanitize_filename("...hidden.txt")
        assert not result.startswith(".")

    def test_trailing_dot_stripped(self):
        result = sanitize_filename("file.txt.")
        assert not result.endswith(".")

    # ── length limit ─────────────────────────────────────────────────────────

    def test_long_filename_truncated_to_255(self):
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_extension_preserved_after_truncation(self):
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert result.endswith(".pdf")

    # ── Windows reserved names ───────────────────────────────────────────────

    def test_con_prefixed(self):
        result = sanitize_filename("CON.txt")
        assert not result.upper().startswith("CON.")
        # The file must still be usable — prefixed, not deleted
        assert result != ""

    def test_nul_prefixed(self):
        result = sanitize_filename("NUL")
        assert result.upper() != "NUL"

    def test_com1_prefixed(self):
        result = sanitize_filename("COM1.log")
        assert not result.upper().startswith("COM1.")

    def test_lpt9_prefixed(self):
        result = sanitize_filename("LPT9.txt")
        assert not result.upper().startswith("LPT9.")

    # ── error cases ──────────────────────────────────────────────────────────

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_filename("")

    def test_only_dots_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_filename("...")

    def test_only_unsafe_chars_raises_value_error(self):
        # All chars get stripped → empty result → ValueError
        with pytest.raises(ValueError):
            sanitize_filename("\x00\x01\x02")


# ===========================================================================
# TODO: Test sanitize_notebook_name
# ===========================================================================

class TestSanitizeNotebookName:

    # ── happy path ──────────────────────────────────────────────────────────

    def test_normal_name_unchanged(self):
        assert sanitize_notebook_name("My Notebook") == "My Notebook"

    def test_hyphens_allowed(self):
        assert sanitize_notebook_name("Study-Notes") == "Study-Notes"

    def test_numbers_allowed(self):
        assert sanitize_notebook_name("Chapter 1") == "Chapter 1"

    def test_mixed_case_preserved(self):
        assert sanitize_notebook_name("UPPER lower") == "UPPER lower"

    def test_returns_string(self):
        assert isinstance(sanitize_notebook_name("NB"), str)

    # ── whitespace handling ──────────────────────────────────────────────────

    def test_leading_trailing_spaces_stripped(self):
        assert sanitize_notebook_name("  hello  ") == "hello"

    def test_internal_multiple_spaces_collapsed(self):
        assert sanitize_notebook_name("a   b") == "a b"

    def test_tabs_stripped(self):
        # Tabs are not in the allowed set → removed, then spaces collapsed
        result = sanitize_notebook_name("a\tb")
        assert "\t" not in result

    # ── character filtering ──────────────────────────────────────────────────

    def test_special_chars_removed(self):
        result = sanitize_notebook_name("NB! @#$%")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_unicode_symbols_removed(self):
        result = sanitize_notebook_name("NB★☆")
        assert "★" not in result

    def test_dots_removed(self):
        # Dots are not in [a-zA-Z0-9 -]
        result = sanitize_notebook_name("v1.0 notes")
        assert "." not in result

    def test_underscores_removed(self):
        result = sanitize_notebook_name("my_notebook")
        assert "_" not in result

    def test_only_allowed_chars_in_result(self):
        name = "Hello World-2024"
        result = sanitize_notebook_name(name)
        assert re.match(r"^[a-zA-Z0-9 \-]+$", result)

    # ── length limit ─────────────────────────────────────────────────────────

    def test_exactly_100_chars_allowed(self):
        name = "a" * 100
        result = sanitize_notebook_name(name)
        assert len(result) == 100

    def test_over_100_chars_truncated(self):
        name = "a" * 150
        result = sanitize_notebook_name(name)
        assert len(result) <= 100

    def test_truncation_does_not_leave_trailing_space(self):
        # Put a space at position 100 to test re-strip after truncation
        name = "a" * 99 + " " + "b" * 50
        result = sanitize_notebook_name(name)
        assert not result.endswith(" ")

    # ── error cases ──────────────────────────────────────────────────────────

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_notebook_name("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_notebook_name("   ")

    def test_all_special_chars_raises_value_error(self):
        # Every character gets stripped → empty → ValueError
        with pytest.raises(ValueError):
            sanitize_notebook_name("!@#$%^&*()")

    def test_only_dots_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_notebook_name("...")


# ===========================================================================
# TODO: Test sanitize_username
# ===========================================================================

class TestSanitizeUsername:

    # ── happy path ──────────────────────────────────────────────────────────

    def test_simple_username_unchanged(self):
        assert sanitize_username("alice") == "alice"

    def test_hyphens_preserved(self):
        assert sanitize_username("my-user") == "my-user"

    def test_numbers_preserved(self):
        assert sanitize_username("user42") == "user42"

    def test_returns_string(self):
        assert isinstance(sanitize_username("alice"), str)

    # ── case folding ─────────────────────────────────────────────────────────

    def test_uppercase_lowercased(self):
        assert sanitize_username("Alice") == "alice"

    def test_mixed_case_lowercased(self):
        assert sanitize_username("AliceSmith") == "alicesmith"

    def test_all_caps_lowercased(self):
        assert sanitize_username("ADMIN") == "admin"

    # ── character replacement ────────────────────────────────────────────────

    def test_dots_replaced_with_underscore(self):
        result = sanitize_username("user.name")
        assert "." not in result
        assert "_" in result

    def test_at_sign_replaced(self):
        result = sanitize_username("user@hf.co")
        assert "@" not in result

    def test_spaces_replaced(self):
        result = sanitize_username("john doe")
        assert " " not in result

    def test_slash_replaced(self):
        result = sanitize_username("org/repo")
        assert "/" not in result

    def test_only_safe_chars_in_result(self):
        result = sanitize_username("My.User@Name!")
        assert re.match(r"^[a-z0-9\-_]+$", result)

    # ── length limit ─────────────────────────────────────────────────────────

    def test_long_username_truncated(self):
        result = sanitize_username("a" * 200)
        assert len(result) <= 64

    def test_exactly_64_chars_allowed(self):
        name = "a" * 64
        result = sanitize_username(name)
        assert len(result) == 64

    # ── leading/trailing whitespace ──────────────────────────────────────────

    def test_leading_spaces_stripped(self):
        result = sanitize_username("  alice")
        assert not result.startswith(" ")

    def test_trailing_spaces_stripped(self):
        result = sanitize_username("alice  ")
        assert not result.endswith(" ")

    # ── error cases ──────────────────────────────────────────────────────────

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_username("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError):
            sanitize_username("   ")


# ===========================================================================
# TODO: Test validate_path (happy path + path traversal rejection)
# ===========================================================================

class TestValidatePath:

    # ── happy path ──────────────────────────────────────────────────────────

    def test_direct_child_allowed(self, tmp_path):
        child = tmp_path / "notebooks"
        result = validate_path(str(child), str(tmp_path))
        assert isinstance(result, Path)

    def test_deep_descendant_allowed(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "file.txt"
        result = validate_path(str(deep), str(tmp_path))
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_root_itself_allowed(self, tmp_path):
        # The root itself is inside itself
        result = validate_path(str(tmp_path), str(tmp_path))
        assert result == tmp_path.resolve()

    def test_returns_path_object(self, tmp_path):
        child = tmp_path / "sub"
        result = validate_path(str(child), str(tmp_path))
        assert isinstance(result, Path)

    def test_returns_resolved_path(self, tmp_path):
        # Path with a redundant dot — resolve() removes it
        child = str(tmp_path) + "/./sub"
        result = validate_path(child, str(tmp_path))
        assert ".." not in str(result)
        assert "/." not in str(result)

    # ── path traversal rejection ─────────────────────────────────────────────

    def test_double_dot_traversal_rejected(self, tmp_path):
        traversal = str(tmp_path) + "/../../etc/passwd"
        with pytest.raises(ValueError):
            validate_path(traversal, str(tmp_path))

    def test_sibling_directory_rejected(self, tmp_path):
        # /tmp/pytest-xxx/sibling — a different directory at the same level
        sibling = tmp_path.parent / "sibling_dir"
        with pytest.raises(ValueError):
            validate_path(str(sibling), str(tmp_path))

    def test_parent_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            validate_path(str(tmp_path.parent), str(tmp_path))

    def test_absolute_etc_passwd_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            validate_path("/etc/passwd", str(tmp_path))

    def test_absolute_root_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            validate_path("/", str(tmp_path))

    def test_raises_value_error_not_other(self, tmp_path):
        # Must be ValueError specifically (callers catch only ValueError)
        traversal = str(tmp_path) + "/../../etc/passwd"
        with pytest.raises(ValueError):
            validate_path(traversal, str(tmp_path))

    def test_error_message_is_descriptive(self, tmp_path):
        traversal = str(tmp_path) + "/../escape"
        with pytest.raises(ValueError, match="outside"):
            validate_path(traversal, str(tmp_path))

    # ── symlink safety ───────────────────────────────────────────────────────

    def test_symlink_resolved_before_check(self, tmp_path):
        # Create a symlink pointing outside tmp_path
        import os
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "evil_link"
        try:
            os.symlink(str(outside), str(link))
            with pytest.raises(ValueError):
                validate_path(str(link), str(tmp_path))
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")
        finally:
            if link.exists() or link.is_symlink():
                link.unlink()
            if outside.exists():
                outside.rmdir()


# ===========================================================================
# TODO: Test generate_notebook_id (format, uniqueness)
# ===========================================================================

class TestGenerateNotebookId:

    # ── format ───────────────────────────────────────────────────────────────

    def test_returns_string(self):
        assert isinstance(generate_notebook_id(), str)

    def test_matches_uuid4_format(self):
        uid = generate_notebook_id()
        assert UUID4_RE.match(uid), f"Not a valid UUID4: {uid!r}"

    def test_correct_length(self):
        # UUID4 string is always 36 characters (32 hex + 4 hyphens)
        assert len(generate_notebook_id()) == 36

    def test_version_digit_is_4(self):
        uid = generate_notebook_id()
        # 15th character (index 14) must be '4' for UUID version 4
        assert uid[14] == "4"

    def test_variant_bits_correct(self):
        uid = generate_notebook_id()
        # 20th character (index 19) must be 8, 9, a, or b (RFC 4122 variant)
        assert uid[19] in "89ab"

    def test_only_hex_and_hyphens(self):
        uid = generate_notebook_id()
        assert re.match(r"^[0-9a-f\-]+$", uid)

    def test_hyphens_at_correct_positions(self):
        uid = generate_notebook_id()
        assert uid[8] == "-"
        assert uid[13] == "-"
        assert uid[18] == "-"
        assert uid[23] == "-"

    # ── uniqueness ───────────────────────────────────────────────────────────

    def test_two_calls_differ(self):
        assert generate_notebook_id() != generate_notebook_id()

    def test_large_batch_all_unique(self):
        ids = [generate_notebook_id() for _ in range(1000)]
        assert len(set(ids)) == 1000

    def test_no_leading_trailing_whitespace(self):
        uid = generate_notebook_id()
        assert uid == uid.strip()

    def test_all_lowercase(self):
        uid = generate_notebook_id()
        assert uid == uid.lower()


# ===========================================================================
# Bonus: Test safe_path (uses validate_path internally)
# ===========================================================================

class TestSafePath:

    def test_normal_subpath_allowed(self, tmp_path):
        result = safe_path(tmp_path, "users", "alice", "notebooks")
        assert result.is_relative_to(tmp_path.resolve())

    def test_returns_path_object(self, tmp_path):
        result = safe_path(tmp_path, "sub")
        assert isinstance(result, Path)

    def test_traversal_via_parts_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            safe_path(tmp_path, "..", "etc", "passwd")

    def test_single_part(self, tmp_path):
        result = safe_path(tmp_path, "notebooks")
        assert result == (tmp_path / "notebooks").resolve()

    def test_multiple_parts_joined(self, tmp_path):
        result = safe_path(tmp_path, "a", "b", "c")
        assert result == (tmp_path / "a" / "b" / "c").resolve()