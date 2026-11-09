"""
LLM Answer Generator using Google Gemini
========================================
Explanation for Viva:
1. Grounded Generation (RAG):
   - Rather than asking the LLM from its pre-trained memory (which can hallucinate),
     we inject ONLY the retrieved chunks into the prompt as the reference context.
   - The LLM is strictly instructed to answer ONLY from the provided material.
2. Similarity Threshold Guardrail:
   - If the top similarity score is less than SIMILARITY_THRESHOLD (from config/settings),
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
from typing import List, Optional
from google import genai

from backend.app.config import settings
from backend.app.models import Chunk

# Predefined exact fallback message in professional English
NOT_FOUND_MESSAGE = "I'm sorry, I couldn't find the answer to this question in the provided documents."

MODEL_NAME = "gemini-3.5-flash"


class GeminiAnswerGenerator:
    def __init__(self, api_key: Optional[str] = None, threshold: Optional[float] = None, model: str = MODEL_NAME):
        """
        Initialize the Gemini generator.
        Reads API key from .env (via settings.GEMINI_API_KEY or os.getenv).
        """
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def generate_answer(
        self,
        query: str,
        chunks: List[Chunk],
        threshold: Optional[float] = None,
        override_threshold: Optional[float] = None,
    ) -> str:
        """
        Generate grounded answer from retrieved chunks.

        Rules:
        1. If chunks is empty -> return NOT_FOUND_MESSAGE
        2. If top similarity < threshold -> return NOT_FOUND_MESSAGE
        3. Else -> prompt Gemini with strictly retrieved chunks and return the answer.
        """
        # Ensure threshold passed as argument is used, falling back to self.threshold
        actual_threshold = (
            threshold
            if threshold is not None
            else (override_threshold if override_threshold is not None else self.threshold)
        )
        threshold = actual_threshold

        # Rule 1: No chunks retrieved
        if not chunks:
            return NOT_FOUND_MESSAGE

        # Rule 2: Top similarity score check
        valid_scores = [
            getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', getattr(c, 'similarity_score', None))))
            for c in chunks
        ]
        valid_scores = [s for s in valid_scores if s is not None]
        top_similarity = max(valid_scores) if valid_scores else 0.0

        print(f"[DEBUG LLM] Chunks received: {len(chunks)}")
        print(f"[DEBUG LLM] Scores found: {[getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', None))) for c in chunks]}")
        print(f"[DEBUG LLM] Top Similarity: {top_similarity} | Threshold: {threshold}")
        print(f"[Threshold Check] Top: {top_similarity:.4f}, Threshold: {threshold:.4f}")

        if top_similarity < threshold:
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

--- REFERENCE MATERIAL ---
{context_text}  
--------------------------

STUDENT QUESTION:
{query}

ANSWER:"""

        t_gemini_start = time.time()
        max_retries = 3
        delay = 5

        try:
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                    err_str = str(e).upper()
                    is_retryable = (
                        "429" in err_str
                        or "503" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "UNAVAILABLE" in err_str
                    )

                    if is_retryable and attempt < max_retries:
                        print(
                            f"[Gemini Retry] Received 429/503 ({e}). "
                            f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(delay)
                        continue

                    # If non-retryable error or all retries exhausted, print and return error
                    print(f"[Gemini Error] API call failed: {e}")
                    return f"Error: {e}"
        finally:
            gemini_duration_ms = round((time.time() - t_gemini_start) * 1000, 2)
            print(f"[Timing LLM] Gemini total time: {gemini_duration_ms}ms")
