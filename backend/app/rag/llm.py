"""
LLM Answer Generator using Google Gemini
========================================
Explanation for Viva:
1. Grounded Generation (RAG):
   - Rather than asking the LLM from its pre-trained memory (which can hallucinate),
     we inject ONLY the retrieved chunks into the prompt as the reference context.
   - The LLM is strictly instructed to answer ONLY from the provided material.
2. Similarity Threshold Guardrail:
   - If the retrieved chunks fail the confidence guardrail (evaluate() in rag/guardrail.py),
     it means the uploaded documents do not contain relevant content for this question.
   - In that case, we do NOT call the LLM and immediately return:
     "I'm sorry, I couldn't find the answer to this question in the provided documents."
   - This prevents hallucinations and saves API credits.
3. System Prompt & Citations:
   - Chunks are numbered and tagged with their page numbers ([Page X]: ...).
   - The LLM is instructed to reference page numbers in its explanation.
"""

import os
import time
import concurrent.futures
from typing import List, Optional
from google import genai
from google.genai import types

from backend.app.config import settings
from backend.app.models import Chunk
from backend.app.rag.guardrail import evaluate as guardrail_evaluate

# Predefined exact fallback message in professional English
NOT_FOUND_MESSAGE = "I'm sorry, I couldn't find the answer to this question in the provided documents."
LLM_UNAVAILABLE_MESSAGE = (
    "The language model is temporarily unavailable (quota or network). "
    "Here are the most relevant passages from your notes."
)

# Persistent executor pool with non-blocking timeout handling (avoids shutdown(wait=True) blocking)
_LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class GeminiAnswerGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        threshold: Optional[float] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Gemini generator.
        Reads API key and model from config/settings or environment variables.
        """
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
        self.model = (
            model
            or getattr(settings, "GEMINI_MODEL", None)
            or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        )
        http_options = types.HttpOptions(
            timeout=5000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key, http_options=http_options)
            except Exception as e:
                print(f"[Gemini Init Warning] Failed to initialize genai.Client: {e}")
                self.client = None
        else:
            self.client = None

    def generate_answer(
        self,
        query: str,
        chunks: List[Chunk],
        threshold: Optional[float] = None,
        override_threshold: Optional[float] = None,
        mode: str = "hybrid",
    ) -> str:
        """
        Generate grounded answer from retrieved chunks.

        Rules:
        1. If chunks is empty -> return NOT_FOUND_MESSAGE
        2. Evaluate via guardrail (vector requires dense similarity >= threshold;
           hybrid allows dense >= threshold or keyword coverage >= 60% with >= 2 words)
        3. Else -> prompt Gemini with strictly retrieved chunks and return the answer.
        """
        # Ensure threshold passed as argument is used, falling back to self.threshold
        actual_threshold = (
            threshold
            if threshold is not None
            else (override_threshold if override_threshold is not None else self.threshold)
        )
        threshold = actual_threshold

        # Rule 1 & 2: Guardrail check
        guardrail_result = guardrail_evaluate(
            chunks=chunks,
            query=query,
            threshold=threshold,
            mode=mode,
        )
        if not guardrail_result.passed:
            return NOT_FOUND_MESSAGE

        # Rule 3: Build grounded context prompt
        context_parts = []
        for idx, chunk in enumerate(chunks, start=1):
            page_num = chunk.metadata.page_number
            subj = chunk.metadata.subject
            unit = chunk.metadata.unit
            context_parts.append(
                f"[Source Chunk {idx} | Subject: {subj} | Unit: {unit} | Page: {page_num}]\n{chunk.text}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful and precise university academic tutor assisting a student with their course syllabus.

IMPORTANT RULES:
1. Answer the student's question STRICTLY and ONLY using the provided reference material below.
2. Do NOT extrapolate or assume facts not present in the reference material.
3. If the reference material does not contain enough information to answer the question, you MUST respond exactly with:
   "{NOT_FOUND_MESSAGE}"
4. When explaining the answer, cite the page number(s) (e.g., "[Page {chunks[0].metadata.page_number}]") so the student can verify it in their textbook/notes.
5. Keep your tone academic, clear, and easy to understand for viva preparation.
6. Keep your answer concise and focused (around 250 words), suitable for university viva preparation.

--- REFERENCE MATERIAL ---
{context_text}  
--------------------------

STUDENT QUESTION:
{query}

ANSWER:"""

        # Offline Mock Mode support for testing when external API quota is unavailable
        if getattr(settings, "LLM_PROVIDER", "").lower() == "mock" or os.getenv("MOCK_LLM", "false").lower() == "true":
            snippets = [f"[Page {c.metadata.page_number}]: {c.text[:220].strip()}..." for c in chunks[:3]]
            return (
                f"**Grounded Synthesis (Offline Mock):**\n\n"
                f"From your course syllabus reference material regarding '{query}':\n\n"
                + "\n\n".join(f"- {s}" for s in snippets)
            )

        if not self.client or not self.api_key:
            return LLM_UNAVAILABLE_MESSAGE

        t_gemini_start = time.time()

        def _call_gemini():
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=1500),
            ).text

        try:
            future = _LLM_EXECUTOR.submit(_call_gemini)
            return future.result(timeout=4.0)
        except concurrent.futures.TimeoutError:
            print("[Gemini Timeout] LLM call exceeded 4.0 seconds. Returning grounded fallback context.")
            return LLM_UNAVAILABLE_MESSAGE
        except Exception as e:
            # When quota is exhausted (429 RESOURCE_EXHAUSTED) or network drops,
            # fail fast and return graceful fallback so the student immediately sees their citations!
            print(f"[Gemini Error] API call failed: {e}")
            return LLM_UNAVAILABLE_MESSAGE
        finally:
            gemini_duration_ms = round((time.time() - t_gemini_start) * 1000, 2)
            print(f"[Timing LLM] Gemini total time: {gemini_duration_ms}ms")
