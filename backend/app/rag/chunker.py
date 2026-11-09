import re
from typing import List, Dict, Any, Optional
import tiktoken
from backend.app.models import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Roman numeral -> integer helper
# ---------------------------------------------------------------------------
_ROMAN_MAP = [
    ("M",  1000), ("CM", 900), ("D",  500), ("CD", 400),
    ("C",  100),  ("XC", 90),  ("L",  50),  ("XL", 40),
    ("X",  10),   ("IX", 9),   ("V",  5),   ("IV", 4),
    ("I",  1),
]

def _roman_to_int(s: str) -> int:
    """Convert an uppercase Roman numeral string to an integer (min 1)."""
    s = s.upper()
    result, i = 0, 0
    for numeral, value in _ROMAN_MAP:
        while s[i:i + len(numeral)] == numeral:
            result += value
            i += len(numeral)
    return result if result > 0 else 1


# Precompiled unit-heading pattern: UNIT[\s\-:]*([IVX]+|\d+) (case-insensitive)
_UNIT_RE = re.compile(r"UNIT[\s\-:]*([IVX]+|\d+)\b", re.IGNORECASE)


def _detect_unit_in_text(text: str) -> Optional[int]:
    """
    Return the integer of the LAST unit heading found in text, or None.
    Handles Roman numerals (I, II, III, IV ...) and plain digits (1, 2 ...).
    """
    last_unit: Optional[int] = None
    for m in _UNIT_RE.finditer(text):
        raw = m.group(1).upper()
        last_unit = int(raw) if raw.isdigit() else _roman_to_int(raw)
    return last_unit


class TokenChunker:
    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 50, encoding_name: str = "cl100k_base"):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        return max(1, int(len(text.split()) * 1.33))

    def _encode(self, text: str) -> List[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text, disallowed_special=())
        return [hash(w) for w in text.split()]

    def _decode(self, tokens: List[int], original_text: str = "") -> str:
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        return original_text

    def chunk_page_text(
        self,
        page_text: str,
        page_number: int,
        doc_id: str,
        filename: str,
        subject: str,
        carried_unit: Optional[int] = None,
    ):
        """
        Chunks the text of a single page to ~target_tokens with overlap.
        Detects headings with regex UNIT[\s\-:]*([IVX]+|\d+) (case-insensitive).
        Carries current unit forward across chunks until the next heading.
        Returns (List[Chunk], Optional[int]).
        """
        if not page_text.strip():
            return [], carried_unit

        chunks: List[Chunk] = []
        current_unit = carried_unit

        if self.tokenizer:
            tokens = self.tokenizer.encode(page_text, disallowed_special=())
            total_tokens = len(tokens)

            if total_tokens <= self.target_tokens:
                sub_unit = _detect_unit_in_text(page_text)
                if sub_unit is not None:
                    current_unit = sub_unit

                chunk_id = f"{doc_id}_p{page_number}_c0"
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    subject=subject,
                    unit=current_unit,
                    page_number=page_number,
                    char_start=0,
                    char_end=len(page_text),
                    token_count=total_tokens,
                )
                return [Chunk(chunk_id=chunk_id, text=page_text, metadata=meta)], current_unit

            step = max(1, self.target_tokens - self.overlap_tokens)
            chunk_idx = 0
            start = 0

            while start < total_tokens:
                end = min(start + self.target_tokens, total_tokens)
                chunk_token_slice = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_token_slice).strip()

                if chunk_text:
                    sub_unit = _detect_unit_in_text(chunk_text)
                    if sub_unit is not None:
                        current_unit = sub_unit

                    chunk_id = f"{doc_id}_p{page_number}_c{chunk_idx}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        subject=subject,
                        unit=current_unit,
                        page_number=page_number,
                        char_start=0,
                        char_end=len(chunk_text),
                        token_count=len(chunk_token_slice),
                    )
                    chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))
                    chunk_idx += 1

                if end >= total_tokens:
                    break
                start += step

        else:
            # Word-based fallback
            words = page_text.split()
            word_target = int(self.target_tokens * 0.75)
            word_overlap = int(self.overlap_tokens * 0.75)
            step = max(1, word_target - word_overlap)
            chunk_idx = 0
            start = 0

            while start < len(words):
                end = min(start + word_target, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words).strip()

                if chunk_text:
                    sub_unit = _detect_unit_in_text(chunk_text)
                    if sub_unit is not None:
                        current_unit = sub_unit

                    chunk_id = f"{doc_id}_p{page_number}_c{chunk_idx}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        subject=subject,
                        unit=current_unit,
                        page_number=page_number,
                        token_count=int(len(chunk_words) * 1.33),
                    )
                    chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))
                    chunk_idx += 1

                if end >= len(words):
                    break
                start += step

        return chunks, current_unit

    def process_document(
        self,
        pages_data: List[Dict[str, Any]],
        doc_id: str,
        filename: str,
        subject: str,
        unit: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Process all pages. Carries current unit forward across chunks and pages.
        Chunks before the first heading get no unit (unit=None).
        PDFs without UNIT headings get no unit (unit=None).
        """
        all_chunks: List[Chunk] = []
        current_unit: Optional[int] = None

        for page_info in pages_data:
            page_text = page_info["text"]
            page_chunks, current_unit = self.chunk_page_text(
                page_text=page_text,
                page_number=page_info["page_number"],
                doc_id=doc_id,
                filename=filename,
                subject=subject,
                carried_unit=current_unit,
            )
            all_chunks.extend(page_chunks)

        return all_chunks
