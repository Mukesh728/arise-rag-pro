"""
Hybrid Search Module
====================
Explanation for Viva:
1. Why Hybrid Search?
   - Vector Search (Dense Retrieval): Understands conceptual meaning and semantic context
     (e.g., "how processes communicate" matches "IPC mechanisms").
   - BM25 Search (Sparse Retrieval): Excels at exact keyword matches, technical terms,
     acronyms, and definitions (e.g., "SJF algorithm", "LRU", "mutex lock").
   - Combining both produces substantially higher retrieval accuracy than either alone.

2. What is Reciprocal Rank Fusion (RRF)?
   - Vector scores are bounded (e.g., 0 to 1), whereas BM25 scores are unbounded positive numbers.
   - We cannot simply add raw scores together.
   - RRF solves this by ranking documents by their position, independent of score scales:
     RRF_Score(d) = sum( 1.0 / (RRF_K + rank(d)) )
     where RRF_K is a smoothing constant (typically 60).
   - Chunks that rank well in BOTH vector search and BM25 get the highest combined score.

3. Mode parameter:
   - "vector": Dense vector search only.
   - "hybrid": Dense vector search + BM25 keyword search merged via RRF.
"""

from typing import List, Optional, Dict
from backend.app.config import settings
from backend.app.models import Chunk
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index


class HybridSearchService:
    def __init__(self, vector_store: ChromaVectorStore, bm25_index: BM25Index):
        """
        Initialize with references to Chroma vector store and BM25 index.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = settings.RRF_K

    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        unit: Optional[str] = None,
        mode: str = "hybrid",
        top_k: int = 3
    ) -> List[Chunk]:
        """
        Perform search using either 'vector' or 'hybrid' mode.

        Args:
            query: The user's question or search query.
            subject: Optional subject filter.
            unit: Optional unit filter.
            mode: "vector" or "hybrid".
            top_k: Number of top chunks to return.

        Returns:
            List of ranked Chunk objects.
        """
        mode_clean = (mode or "hybrid").strip().lower()

        # Mode 1: Pure Vector Search
        if mode_clean == "vector":
            return self._vector_search(query=query, subject=subject, unit=unit, top_k=top_k)

        # Mode 2: Hybrid Search (Vector + BM25 fused with RRF)
        return self._hybrid_rrf_search(query=query, subject=subject, unit=unit, top_k=top_k)

    def _vector_search(
        self,
        query: str,
        subject: Optional[str] = None,
        unit: Optional[str] = None,
        top_k: int = 3
    ) -> List[Chunk]:
        """Runs vector search only and assigns ranks."""
        vector_results = self.vector_store.search(
            query=query,
            subject=subject,
            unit=unit,
            top_k=top_k
        )

        ranked_chunks: List[Chunk] = []
        for rank_idx, (chunk, score) in enumerate(vector_results):
            chunk.vector_score = score
            chunk.score = score
            chunk.similarity = score
            chunk.similarity_score = score
            chunk.rank = rank_idx + 1
            ranked_chunks.append(chunk)

        return ranked_chunks

    def _hybrid_rrf_search(
        self,
        query: str,
        subject: Optional[str] = None,
        unit: Optional[str] = None,
        top_k: int = 3
    ) -> List[Chunk]:
        """
        Executes both Vector Search and BM25 Search, then merges candidate lists
        using Reciprocal Rank Fusion (RRF).
        """
        # Fetch more candidates for fusion to allow overlap discovery (e.g. 2x to 3x top_k)
        candidate_k = max(top_k * 3, 10)

        # 1. Retrieve candidates from Vector Store
        vector_candidates = self.vector_store.search(
            query=query,
            subject=subject,
            unit=unit,
            top_k=candidate_k
        )

        # 2. Retrieve candidates from BM25 Keyword Search
        bm25_candidates = self.bm25_index.search(
            query=query,
            subject=subject,
            unit=unit,
            top_k=candidate_k
        )

        # If both are empty, return nothing
        if not vector_candidates and not bm25_candidates:
            return []

        # If only vector candidates exist
        if vector_candidates and not bm25_candidates:
            return self._vector_search(query=query, subject=subject, unit=unit, top_k=top_k)

        # If only BM25 candidates exist
        if bm25_candidates and not vector_candidates:
            ranked_chunks: List[Chunk] = []
            for rank_idx, (chunk, bm25_score) in enumerate(bm25_candidates[:top_k]):
                chunk.bm25_score = bm25_score
                chunk.rank = rank_idx + 1
                ranked_chunks.append(chunk)
            return ranked_chunks

        # 3. Perform Reciprocal Rank Fusion
        # Mapping from chunk_id -> Chunk object
        chunk_map: Dict[str, Chunk] = {}
        # Mapping from chunk_id -> cumulative RRF score
        rrf_scores: Dict[str, float] = {}

        # Process Vector candidates
        for rank_idx, (chunk, sim_score) in enumerate(vector_candidates):
            c_id = chunk.chunk_id
            chunk.vector_score = sim_score
            chunk.score = sim_score
            chunk.similarity = sim_score
            chunk.similarity_score = sim_score
            chunk_map[c_id] = chunk

            # RRF formula: 1.0 / (k + rank)
            # rank is 1-indexed (rank_idx + 1)
            rank = rank_idx + 1
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Process BM25 candidates
        for rank_idx, (chunk, bm25_score) in enumerate(bm25_candidates):
            c_id = chunk.chunk_id
            rank = rank_idx + 1

            if c_id in chunk_map:
                # Chunk was already retrieved by vector search
                chunk_map[c_id].bm25_score = bm25_score
            else:
                chunk.bm25_score = bm25_score
                chunk_map[c_id] = chunk

            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 4. Sort chunks descending by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_chunks: List[Chunk] = []
        for overall_rank, cid in enumerate(sorted_ids[:top_k], start=1):
            c = chunk_map[cid]
            c.rrf_score = round(rrf_scores[cid], 6)
            c.rank = overall_rank
            final_chunks.append(c)

        return final_chunks
