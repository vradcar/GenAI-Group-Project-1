"""
Artifact Generator.

Responsibilities:
  - Gather all extracted texts from the notebook
  - If combined text exceeds context window, use map-reduce summarization
  - Generate artifacts via LLM with artifact-specific prompts:
      * Report  — structured summary with headings and source refs (.md)
      * Quiz    — N questions (MCQ / short-answer / T-F) + answer key (.md)
      * Podcast — conversational transcript (.md) + TTS audio (.mp3)
  - Save outputs to artifacts/<type>/
"""

import logging
from pathlib import Path
from datetime import datetime

from core import llm_client
from storage.notebook_store import get_artifact_dir

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
    """Generate a podcast transcript and synthesize to MP3 via edge-tts.

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
        
        # Generate podcast transcript via LLM
        prompt = f"""Create a conversational podcast transcript about the following material.
Write as if two knowledgeable hosts are discussing the topic naturally.
Include:
- Introduction (hook the listener)
- Main discussion (key concepts explained conversationally)
- Key takeaways
- Conclusion

Keep it engaging and accessible. Target length: 5-10 minutes of speech.

Material:
{notebook_text}"""

        response = llm_client.complete(
            prompt=prompt,
            system_prompt="You are a podcast scriptwriter. Write engaging, conversational dialogue for two hosts. Use Markdown format with [Host 1] and [Host 2] labels.",
            temperature=0.6,
        )
        
        transcript = response.text
        
        # Save transcript
        artifact_dir = get_artifact_dir(username, notebook_id, "podcasts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = artifact_dir / f"transcript_{timestamp}.md"
        transcript_file.write_text(transcript, encoding="utf-8")
        
        # Synthesize to speech using edge-tts
        audio_file = artifact_dir / f"podcast_{timestamp}.mp3"
        
        # Extract clean text for TTS (remove markdown formatting)
        clean_text = transcript.replace("[Host 1]", "").replace("[Host 2]", "").strip()
        clean_text = clean_text[:3000]  # Limit to ~5 min of audio
        
        async def synthesize():
            communicate = edge_tts.Communicate(text=clean_text, voice="en-US-AriaNeural")
            await communicate.save(str(audio_file))
        
        try:
            asyncio.run(synthesize())
            logger.info(f"Podcast audio saved to {audio_file}")
        except Exception as tts_error:
            logger.warning(f"TTS synthesis failed: {tts_error}. Saving transcript only.")
            # Return transcript path but no audio
            return str(transcript_file), None
        
        logger.info(f"Podcast created: transcript={transcript_file}, audio={audio_file}")
        return str(transcript_file), str(audio_file)
        
    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        return None, None
