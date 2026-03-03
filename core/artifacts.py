"""
Artifact Generator.

Responsibilities:
  - Gather all extracted texts from the notebook
  - If combined text exceeds context window, use map-reduce summarization
  - Generate artifacts via LLM with artifact-specific prompts:
      * Report  — structured summary with headings and source refs (.md)
      * Quiz    — N questions (MCQ / short-answer / T-F) + answer key (.md)
      * Podcast — two-speaker conversational transcript (.md) + TTS audio (.mp3)
  - Save outputs to artifacts/<type>/
"""

import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path

from core import llm_client
from storage.notebook_store import get_artifact_dir
from utils.config import (
    TTS_HOST_A_PITCH,
    TTS_HOST_A_RATE,
    TTS_HOST_A_VOICE,
    TTS_HOST_B_PITCH,
    TTS_HOST_B_RATE,
    TTS_HOST_B_VOICE,
    TTS_MAX_CHARS,
)

logger = logging.getLogger(__name__)

# Context window limits (conservative estimates)
MAX_CONTEXT_TOKENS = 8000
AVG_CHARS_PER_TOKEN = 4


def _gather_notebook_text(username: str, notebook_id: str) -> str:
    """Gather all extracted texts from the notebook's vector store."""
    try:
        from storage import vector_store

        # Use the underlying collection to fetch all stored documents and metadatas
        coll = vector_store.get_or_create_collection(username, notebook_id)
        all_data = coll.get(include=["documents", "metadatas"]) or {}
        docs = all_data.get("documents", [])
        # chroma may return nested lists; normalize
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        metadatas = all_data.get("metadatas", [])
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        
        # Combine with source attribution
        combined_text = ""
        for doc, meta in zip(docs, metadatas):
            # metadata field uses 'source' in vector_store ingestion
            source = meta.get("source", meta.get("source_name", "unknown"))
            combined_text += f"\n\n[From {source}]\n{doc}"
        
        return combined_text if combined_text.strip() else ""
    except Exception as e:
        logger.error(f"Error gathering notebook text: {e}")
        return ""


def _should_summarize(text: str) -> bool:
    """Check if text exceeds context window and needs summarization."""
    estimated_tokens = len(text) / AVG_CHARS_PER_TOKEN
    return estimated_tokens > MAX_CONTEXT_TOKENS


def _summarize_text(text: str) -> str:
    """Map-reduce summarization for large texts."""
    if len(text) <= 2000:
        return text
    
    logger.info("Text exceeds context window; applying map-reduce summarization")
    
    # Split into chunks
    chunk_size = 1500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Map: summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        try:
            response = llm_client.complete(
                prompt=f"Summarize this text concisely:\n\n{chunk}",
                system_prompt="You are a concise summarizer. Return only the key points.",
                temperature=0.3,
            )
            chunk_summaries.append(response.text)
        except Exception as e:
            logger.warning(f"Error summarizing chunk {i}: {e}")
            chunk_summaries.append(chunk[:500])
    
    # Reduce: combine summaries
    combined_summary = "\n\n".join(chunk_summaries)
    return combined_summary


def generate_report(username: str, notebook_id: str) -> str:
    """Generate a summary report from all notebook sources."""
    try:
        # Gather text
        notebook_text = _gather_notebook_text(username, notebook_id)
        if not notebook_text.strip():
            return "# Report\n\nNo sources found in this notebook."
        
        # Summarize if needed
        if _should_summarize(notebook_text):
            notebook_text = _summarize_text(notebook_text)
        
        # Generate report via LLM
        prompt = f"""Based on the following material, generate a well-structured report with:
- Executive Summary (2-3 sentences)
- Key Concepts (bullet points)
- Main Topics (with subsections)
- Conclusion

Material:
{notebook_text}"""

        response = llm_client.complete(
            prompt=prompt,
            system_prompt="You are an expert report writer. Generate a professional, well-organized report in Markdown format.",
            temperature=0.5,
        )
        
        report_md = response.text
        
        # Save to artifacts
        artifact_dir = get_artifact_dir(username, notebook_id, "reports")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = artifact_dir / f"report_{timestamp}.md"
        report_file.write_text(report_md, encoding="utf-8")
        
        logger.info(f"Report saved to {report_file}")
        return report_md
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"# Report Generation Error\n\n{str(e)}"


def generate_quiz(username: str, notebook_id: str, num_questions: int = 10) -> str:
    """Generate a quiz with answer key from all notebook sources."""
    try:
        # Gather text
        notebook_text = _gather_notebook_text(username, notebook_id)
        if not notebook_text.strip():
            return "# Quiz\n\nNo sources found to generate quiz."
        
        # Summarize if needed
        if _should_summarize(notebook_text):
            notebook_text = _summarize_text(notebook_text)
        
        # Generate quiz via LLM
        prompt = f"""Based on the following material, generate a quiz with {num_questions} questions.
Include a mix of:
- Multiple choice (MCQ)
- Short answer
- True/False

Format:
## Quiz Questions
[Questions numbered 1-{num_questions}]

## Answer Key
[Answers with brief explanations]

Material:
{notebook_text}"""

        response = llm_client.complete(
            prompt=prompt,
            system_prompt="You are an expert educator. Generate comprehensive quiz questions that test understanding of key concepts. Use Markdown format.",
            temperature=0.5,
        )
        
        quiz_md = response.text
        
        # Save to artifacts
        artifact_dir = get_artifact_dir(username, notebook_id, "quizzes")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quiz_file = artifact_dir / f"quiz_{timestamp}.md"
        quiz_file.write_text(quiz_md, encoding="utf-8")
        
        logger.info(f"Quiz saved to {quiz_file}")
        return quiz_md
        
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return f"# Quiz Generation Error\n\n{str(e)}"


def generate_podcast(username: str, notebook_id: str) -> tuple[str, str]:
    """Generate a two-speaker podcast transcript and synthesize to MP3.

    Each host gets a distinct edge-tts voice + prosody so listeners can
    tell them apart — one upbeat/fast, the other chill/deeper.

    Returns:
        (transcript_path, audio_path)
    """
    try:
        import asyncio

        import edge_tts

        # Gather text
        notebook_text = _gather_notebook_text(username, notebook_id)
        if not notebook_text.strip():
            return None, None

        # Summarize if needed
        if _should_summarize(notebook_text):
            notebook_text = _summarize_text(notebook_text)

        # ── Generate two-speaker transcript via LLM ──────────
        prompt = f"""Create a conversational podcast transcript about the following material.

FORMAT RULES (strict):
- Exactly two speakers: **Alex** (enthusiastic, curious) and **Sam** (thoughtful, laid-back).
- Every line of dialogue MUST start with either  [Alex]:  or  [Sam]:
- No narration or stage directions — dialogue only.
- Alternate speakers often; keep each turn 1-4 sentences.

CONTENT:
1. Cold-open hook (Alex)
2. Main discussion — explain key concepts conversationally
3. Fun analogies or real-world examples
4. Quick-fire takeaways
5. Sign-off

Target length: ~50–70 lines of dialogue (≈5-8 min of audio).

Material:
{notebook_text}"""

        response = llm_client.complete(
            prompt=prompt,
            system_prompt=(
                "You are a top-tier podcast scriptwriter. "
                "Write punchy, engaging dialogue between two hosts: "
                "Alex (energetic, asks great questions) and "
                "Sam (calm, gives clear explanations). "
                "Output ONLY dialogue lines prefixed with [Alex]: or [Sam]:."
            ),
            temperature=0.7,
        )

        transcript = response.text

        # ── Save transcript ──────────────────────────────────
        artifact_dir = get_artifact_dir(username, notebook_id, "podcasts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = artifact_dir / f"transcript_{timestamp}.md"
        transcript_file.write_text(transcript, encoding="utf-8")

        # ── Parse into speaker segments ──────────────────────
        segments = _parse_transcript(transcript)
        if not segments:
            logger.warning("Could not parse speaker segments; falling back to single voice")
            segments = [("A", transcript)]

        # ── Synthesize each segment with the right voice ─────
        audio_file = artifact_dir / f"podcast_{timestamp}.mp3"

        async def _synthesize_two_speakers():
            tmp_paths: list[str] = []
            char_budget = TTS_MAX_CHARS
            try:
                for speaker, text in segments:
                    if char_budget <= 0:
                        break
                    text = text[:char_budget]
                    char_budget -= len(text)

                    if speaker == "B":
                        voice, rate, pitch = TTS_HOST_B_VOICE, TTS_HOST_B_RATE, TTS_HOST_B_PITCH
                    else:
                        voice, rate, pitch = TTS_HOST_A_VOICE, TTS_HOST_A_RATE, TTS_HOST_A_PITCH

                    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    tmp.close()
                    tmp_paths.append(tmp.name)

                    comm = edge_tts.Communicate(
                        text=text, voice=voice, rate=rate, pitch=pitch
                    )
                    await comm.save(tmp.name)

                # Concatenate all segment MP3s
                with open(str(audio_file), "wb") as out:
                    for p in tmp_paths:
                        out.write(Path(p).read_bytes())
            finally:
                for p in tmp_paths:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass

        try:
            asyncio.run(_synthesize_two_speakers())
            logger.info(f"Two-speaker podcast audio saved to {audio_file}")
        except Exception as tts_error:
            logger.warning(f"TTS synthesis failed: {tts_error}. Saving transcript only.")
            return str(transcript_file), None

        logger.info(f"Podcast created: transcript={transcript_file}, audio={audio_file}")
        return str(transcript_file), str(audio_file)

    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        return None, None


# ── Transcript parser ────────────────────────────────────────

# Matches lines like  [Alex]: blah blah  or  **Alex:** blah
_SPEAKER_RE = re.compile(
    r"^\s*(?:\*{0,2})\[?(Alex|Sam)\]?(?:\*{0,2})\s*:\s*(.+)",
    re.IGNORECASE,
)


def _parse_transcript(transcript: str) -> list[tuple[str, str]]:
    """Parse a two-speaker transcript into ``[('A'|'B', text), ...]`` segments.

    Adjacent lines by the same speaker are merged so each TTS call is a
    natural paragraph rather than one sentence.
    """
    segments: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        m = _SPEAKER_RE.match(line)
        if not m:
            # Continuation of previous speaker (or noise) — append if possible
            stripped = line.strip()
            if stripped and segments:
                prev_speaker, prev_text = segments[-1]
                segments[-1] = (prev_speaker, prev_text + " " + stripped)
            continue
        name = m.group(1).lower()
        text = m.group(2).strip()
        speaker = "A" if name == "alex" else "B"
        # Merge with previous if same speaker
        if segments and segments[-1][0] == speaker:
            segments[-1] = (speaker, segments[-1][1] + " " + text)
        else:
            segments.append((speaker, text))
    return segments
