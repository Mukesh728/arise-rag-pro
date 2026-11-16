"""
Retrieval Confidence Guardrail
==============================
Evaluates whether retrieved chunks contain sufficient confidence to generate
an answer, or whether the query should be rejected to prevent hallucination.

Rules:
1. Vector mode:
   Passes ONLY if the maximum dense cosine similarity across retrieved chunks
   is greater than or equal to `threshold`.
2. Hybrid mode:
   Passes if the maximum dense cosine similarity >= `threshold`
   OR the best chunk contains >= 60% of the query's content words
   (with stopwords removed) AND at least 2 matched words.
   This ensures BM25 sparse keyword evidence is respected when dense vector
   scores are absent (e.g., BM25-only fallback) while still rejecting off-topic
   queries (e.g., "Who won the IPL?").
3. Logging:
   Emits one standardized log line per evaluation:
   [Guardrail] mode=... passed=... | reason
"""

import re
import logging
from typing import List, Optional, Set
from backend.app.models import Chunk

logger = logging.getLogger("rag.guardrail")

# Standard English stopwords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}


def get_dense_similarity(c: Chunk) -> Optional[float]:
    """
    Extract dense vector similarity score from Chunk model.
    Checks vector_score, similarity_score, similarity, score in order.
    Returns float or None.
    """
    for val in (c.vector_score, c.similarity_score, c.similarity, c.score):
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return None


def extract_content_words(text: str) -> List[str]:
    """
    Extract non-stopword content words from text.
    Filters out common stopwords and standalone punctuation.
    Preserves digits (e.g. '10') and words longer than 1 character.
    Returns unique content words in order of appearance.
    """
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    words = [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]
    return list(dict.fromkeys(words))


class GuardrailResult:
    """Encapsulates the decision and diagnostics of the retrieval guardrail."""

    def __init__(
        self,
        passed: bool,
        reason: str,
        max_dense_similarity: float,
        best_content_word_ratio: float = 0.0,
        matched_words_count: int = 0,
        total_content_words: int = 0
    ):
        self.passed = passed
        self.reason = reason
        self.max_dense_similarity = max_dense_similarity
        self.best_content_word_ratio = best_content_word_ratio
        self.matched_words_count = matched_words_count
        self.total_content_words = total_content_words

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return f"GuardrailResult(passed={self.passed}, reason='{self.reason}')"


def evaluate(
    chunks: List[Chunk],
    query: str,
    threshold: float,
    mode: str = "hybrid"
) -> GuardrailResult:
    """
    Evaluate retrieval quality against confidence guardrails.

    Args:
        chunks: List of retrieved Chunk objects.
        query: Student question string.
        threshold: Minimum similarity threshold (e.g., 0.32).
        mode: Search strategy ("vector" or "hybrid").

    Returns:
        GuardrailResult object with boolean truthiness and detailed reason.
    """
    mode_clean = (mode or "hybrid").strip().lower()

    if not chunks:
        reason = "no chunks retrieved"
        log_line = f"[Guardrail] mode={mode_clean} passed=False | {reason}"
        print(log_line)
        logger.info(log_line)
        return GuardrailResult(passed=False, reason=reason, max_dense_similarity=0.0)

    # 1. Extract maximum dense similarity
    dense_sims = [get_dense_similarity(c) for c in chunks]
    valid_dense_sims = [s for s in dense_sims if s is not None]
    max_dense_sim = max(valid_dense_sims) if valid_dense_sims else 0.0

    # 2. Vector mode: strict dense similarity check
    if mode_clean == "vector":
        if max_dense_sim >= threshold:
            reason = f"max dense similarity {max_dense_sim:.4f} >= threshold {threshold:.4f}"
            log_line = f"[Guardrail] mode={mode_clean} passed=True | {reason}"
            print(log_line)
            logger.info(log_line)
            return GuardrailResult(passed=True, reason=reason, max_dense_similarity=max_dense_sim)
        else:
            reason = f"max dense similarity {max_dense_sim:.4f} < threshold {threshold:.4f}"
            log_line = f"[Guardrail] mode={mode_clean} passed=False | {reason}"
            print(log_line)
            logger.info(log_line)
            return GuardrailResult(passed=False, reason=reason, max_dense_similarity=max_dense_sim)

    # 3. Hybrid mode: dense similarity OR BM25 content word match
    # First, check if dense similarity passes
    if max_dense_sim >= threshold:
        reason = f"max dense similarity {max_dense_sim:.4f} >= threshold {threshold:.4f}"
        log_line = f"[Guardrail] mode={mode_clean} passed=True | {reason}"
        print(log_line)
        logger.info(log_line)
        return GuardrailResult(passed=True, reason=reason, max_dense_similarity=max_dense_sim)

    # Dense similarity is below threshold or absent (BM25 only):
    # Check if best chunk contains >= 60% of content words AND at least 2 matched words
    content_words = extract_content_words(query)
    total_cw = len(content_words)
    best_ratio = 0.0
    best_count = 0

    if total_cw > 0:
        for c in chunks:
            chunk_tokens = set(re.findall(r'\b[a-zA-Z0-9_]+\b', c.text.lower()))
            matched = [w for w in content_words if w in chunk_tokens]
            count = len(matched)
            ratio = count / total_cw
            if ratio > best_ratio or (ratio == best_ratio and count > best_count):
                best_ratio = ratio
                best_count = count

    keyword_pass = (best_ratio >= 0.60 and best_count >= 2)

    if keyword_pass:
        reason = (
            f"keyword match {best_ratio:.1%} ({best_count}/{total_cw} content words) >= 60% "
            f"with >= 2 words (dense sim: {max_dense_sim:.4f})"
        )
        log_line = f"[Guardrail] mode={mode_clean} passed=True | {reason}"
        print(log_line)
        logger.info(log_line)
        return GuardrailResult(
            passed=True,
            reason=reason,
            max_dense_similarity=max_dense_sim,
            best_content_word_ratio=best_ratio,
            matched_words_count=best_count,
            total_content_words=total_cw
        )
    else:
        reason = (
            f"dense sim {max_dense_sim:.4f} < threshold {threshold:.4f} and keyword match "
            f"{best_ratio:.1%} ({best_count}/{total_cw} words) did not meet 60% with >= 2 words"
        )
        log_line = f"[Guardrail] mode={mode_clean} passed=False | {reason}"
        print(log_line)
        logger.info(log_line)
        return GuardrailResult(
            passed=False,
            reason=reason,
            max_dense_similarity=max_dense_sim,
            best_content_word_ratio=best_ratio,
            matched_words_count=best_count,
            total_content_words=total_cw
        )

