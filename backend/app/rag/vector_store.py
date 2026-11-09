"""
Vector Store Module using ChromaDB
==================================
Explanation for Viva:
1. What is ChromaDB?
   - ChromaDB is an open-source, embedded vector database designed for AI applications.
   - It persists embeddings and document metadata to disk (CHROMA_PERSIST_DIR).
2. What are Embeddings?
   - Text chunks are converted into dense numerical vectors (arrays of floats).
   - Semantically similar texts are placed close together in multi-dimensional space.
3. Distance Metric & Cosine Similarity:
   - We configure HNSW index with 'cosine' space.
   - Chroma returns cosine distance (distance = 1 - cosine_similarity).
   - Therefore: similarity_score = 1.0 - distance.
4. Metadata & Filtering:
   - Each chunk stores metadata: subject, unit, page_number, filename, doc_id.
   - We support metadata filtering (e.g., filter by subject and/or unit) so queries
     only search within the relevant syllabus or uploaded chapter.
"""

import re
from typing import List, Optional, Tuple, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.app.config import settings
from backend.app.models import Chunk, ChunkMetadata


class ChromaVectorStore:
    def __init__(self, collection_name: str = "mrsptu_study_rag"):
        """
        Initialize persistent ChromaDB client and get or create the collection.
        Uses cosine distance so similarity can be easily calculated as (1 - distance).
        """
        # PersistentClient saves data to disk so chunks persist across server restarts
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return total number of chunks currently stored in Chroma."""
        return self.collection.count()

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add or update chunks in Chroma store.
        Each chunk includes text content, unique ID, and metadata (subject, unit, page).
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            
            # Chroma metadata must be primitive types (str, int, float, bool)
            meta_dict = {
                "chunk_id": chunk.metadata.chunk_id,
                "doc_id": chunk.metadata.doc_id,
                "filename": chunk.metadata.filename,
                "subject": chunk.metadata.subject,
                "page": int(chunk.metadata.page_number),
                "page_number": int(chunk.metadata.page_number),
                "token_count": int(chunk.metadata.token_count),
                "char_start": int(chunk.metadata.char_start),
                "char_end": int(chunk.metadata.char_end)
            }
            if chunk.metadata.unit is not None:
                meta_dict["unit"] = int(chunk.metadata.unit)
            metadatas.append(meta_dict)

        # Upsert ensures that existing chunks with the same ID get updated without error
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def _build_where_filter(
        self,
        subject: Optional[str] = None,
        unit: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Builds ChromaDB 'where' dictionary for filtering by subject and/or unit.
        Ignores filters if value is None, empty, or 'all'.
        """
        filters = []

        if subject and subject.strip().lower() != "all":
            filters.append({"subject": subject.strip()})

        if unit is not None:
            unit_str = str(unit).strip().lower()
            if unit_str not in ("", "all"):
                m = re.search(r"\d+", unit_str)
                if m:
                    filters.append({"unit": int(m.group(0))})

        if len(filters) == 1:
            return filters[0]
        elif len(filters) > 1:
            return {"$and": filters}

        return None

    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        unit: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform vector similarity search on Chroma store.
        
        Returns:
            List of tuples: (Chunk object, similarity_score float)
            sorted by similarity score descending.
        """
        total_items = self.collection.count()
        if total_items == 0:
            return []

        # Chroma throws an error if n_results > total count
        n_results = min(top_k, total_items)
        where_filter = self._build_where_filter(subject=subject, unit=unit)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
        except Exception as e:
            # Fallback if where filter finds 0 matches in Chroma
            return []

        chunks_with_scores: List[Tuple[Chunk, float]] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.5] * len(ids)

        for i in range(len(ids)):
            chunk_id = ids[i]
            text = documents[i]
            meta = metadatas[i]
            distance = distances[i]

            # In cosine space: similarity = 1.0 - cosine_distance
            # Clamp between 0.0 and 1.0 to avoid negative or floating precision anomalies
            similarity = max(0.0, min(1.0, 1.0 - distance))
            similarity = round(similarity, 4)

            raw_unit = meta.get("unit")
            unit_val: Optional[int] = None
            if raw_unit is not None:
                try:
                    unit_val = int(raw_unit)
                except (ValueError, TypeError):
                    unit_val = None

            chunk_meta = ChunkMetadata(
                chunk_id=meta.get("chunk_id", chunk_id),
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                subject=meta.get("subject", ""),
                unit=unit_val,
                page_number=int(meta.get("page_number", meta.get("page", 1))),
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", len(text))),
                token_count=int(meta.get("token_count", 0))
            )

            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata=chunk_meta,
                vector_score=similarity,
                score=similarity,
                similarity=similarity,
                similarity_score=similarity,
            )
            chunks_with_scores.append((chunk, similarity))

        # Sort descending by similarity score
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        return chunks_with_scores

    def get_all_chunks(self) -> List[Chunk]:
        """
        Retrieve all stored chunks from Chroma.
        Used at server startup to synchronize the in-memory BM25 index with Chroma.
        """
        total = self.collection.count()
        if total == 0:
            return []

        data = self.collection.get(include=["documents", "metadatas"])
        chunks: List[Chunk] = []

        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""
            raw_unit = meta.get("unit")
            unit_val: Optional[int] = None
            if raw_unit is not None:
                try:
                    unit_val = int(raw_unit)
                except (ValueError, TypeError):
                    unit_val = None

            chunk_meta = ChunkMetadata(
                chunk_id=ids[i],
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                subject=meta.get("subject", ""),
                unit=unit_val,
                page_number=int(meta.get("page_number", meta.get("page", 1))),
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", len(text))),
                token_count=int(meta.get("token_count", 0))
            )
            chunks.append(Chunk(chunk_id=ids[i], text=text, metadata=chunk_meta))

        return chunks

    def get_distinct_subjects_and_units(self) -> Dict[str, Any]:
        """
        Inspect all chunks in the store and return distinct subjects and their units.
        Useful for frontend dropdown selections.
        """
        all_chunks = self.get_all_chunks()
        subjects_set = set()
        units_by_subject: Dict[str, set] = {}

        for c in all_chunks:
            subj = (c.metadata.subject or "").strip()
            if not subj:
                continue
            subjects_set.add(subj)
            if c.metadata.unit is not None:
                try:
                    unit_num = int(c.metadata.unit)
                    if subj not in units_by_subject:
                        units_by_subject[subj] = set()
                    units_by_subject[subj].add(unit_num)
                except (ValueError, TypeError):
                    pass

        clean_units_by_subject: Dict[str, List[str]] = {}
        for s in sorted(list(subjects_set)):
            num_set = units_by_subject.get(s, set())
            sorted_nums = sorted(list(num_set))
            clean_units_by_subject[s] = [f"Unit {n}" for n in sorted_nums]

        return {
            "subjects": sorted(list(subjects_set)),
            "units_by_subject": clean_units_by_subject,
            "total_documents": len(set(c.metadata.doc_id for c in all_chunks if c.metadata.doc_id)),
            "total_chunks": len(all_chunks)
        }

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a particular doc_id."""
        self.collection.delete(where={"doc_id": doc_id})
