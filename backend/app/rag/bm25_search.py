import re
from typing import List, Optional, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from backend.app.models import Chunk

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer with lowercase normalization."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

class BM25Index:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []

    def build_index(self, chunks: List[Chunk]):
        """Build or refresh BM25 index with current chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            self.corpus_tokens = []
            return

        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def add_chunks(self, new_chunks: List[Chunk]):
        """Add new chunks and re-index."""
        self.chunks.extend(new_chunks)
        self.build_index(self.chunks)

    def remove_document(self, doc_id: str):
        """Remove chunks belonging to doc_id and rebuild index."""
        self.chunks = [c for c in self.chunks if c.metadata.doc_id != doc_id]
        self.build_index(self.chunks)

    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        unit: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Search chunks using BM25, respecting subject and unit metadata filters.
        Returns list of (Chunk, bm25_score) tuples sorted descending by score.
        """
        if not self.chunks or self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        candidates: List[Tuple[Chunk, float]] = []
        for i, score in enumerate(scores):
            chunk = self.chunks[i]
            # Metadata filter check
            if subject and subject.strip().lower() != "all":
                if chunk.metadata.subject.strip().lower() != subject.strip().lower():
                    continue
            if unit and str(unit).strip().lower() != "all":
                m_filter = re.search(r"\d+", str(unit).strip())
                if m_filter:
                    filter_num = int(m_filter.group(0))
                    if chunk.metadata.unit != filter_num:
                        continue
                elif str(chunk.metadata.unit or "").strip().lower() != str(unit).strip().lower():
                    continue
            
            if score > 0:
                candidates.append((chunk, float(score)))

        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
