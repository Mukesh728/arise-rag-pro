"""Supabase Postgres/pgvector-backed vector store for Arise.

The public methods intentionally mirror the old Chroma store so the existing
FastAPI routes, BM25 fusion, benchmark, citations, and UI remain unchanged.
Embeddings use Chroma's existing all-MiniLM-L6-v2 DefaultEmbeddingFunction
(384 dimensions); only persistence and nearest-neighbor search move to
Supabase Postgres.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.config import settings
from backend.app.models import Chunk, ChunkMetadata
from backend.app.rag.vector_store import VectorSearchError

logger = logging.getLogger("rag.pgvector_store")

try:
    import psycopg
except ImportError:  # pragma: no cover - dependency installed in deployment
    psycopg = None

try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError:  # pragma: no cover - dependency installed in deployment
    DefaultEmbeddingFunction = None

PgVectorSearchError = VectorSearchError

class SupabasePgVectorStore:
    EMBEDDING_DIM = 384

    def __init__(self, collection_name: str = "arise_chunks"):
        self.collection_name = collection_name
        self._cached_subjects: Optional[Set[str]] = None
        self._embedder = None
        self._ensure_runtime()
        self._ensure_schema()

    def _ensure_runtime(self) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg is required when VECTOR_STORE_TYPE=pgvector")
        if DefaultEmbeddingFunction is None:
            raise RuntimeError("Chroma embedding runtime is required to create 384-dim embeddings")
        if not self._database_url():
            raise RuntimeError("DATABASE_URL must point to the Supabase Postgres connection when VECTOR_STORE_TYPE=pgvector")
        self._embedder = DefaultEmbeddingFunction()

    @staticmethod
    def _database_url() -> str:
        return getattr(settings, "DATABASE_URL", "") or ""

    def _connect(self):
        return psycopg.connect(self._database_url(), connect_timeout=15)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        try:
            values = self._embedder(texts)
            return [list(map(float, row)) for row in values]
        except Exception as exc:
            raise PgVectorSearchError(f"Embedding generation failed: {exc}") from exc

    @staticmethod
    def _vector_literal(values: List[float]) -> str:
        return "[" + ",".join(format(float(v), ".8g") for v in values) + "]"

    @staticmethod
    def _unit_value(unit: Optional[str]) -> Optional[int]:
        if unit is None or str(unit).strip().lower() in ("", "all"):
            return None
        match = re.search(r"\d+", str(unit))
        return int(match.group(0)) if match else None

    @staticmethod
    def _from_row(row: Any) -> Chunk:
        (chunk_id, text, doc_id, filename, subject, unit, page_number,
         char_start, char_end, token_count, owner_user_id, score) = row
        unit_val = int(unit) if unit is not None else None
        similarity = max(0.0, min(1.0, float(score or 0.0)))
        meta = ChunkMetadata(
            chunk_id=str(chunk_id), doc_id=str(doc_id), filename=str(filename or ""),
            subject=str(subject or ""), unit=unit_val,
            page_number=int(page_number or 1), char_start=int(char_start or 0),
            char_end=int(char_end or len(text or "")), token_count=int(token_count or 0),
            owner_user_id=str(owner_user_id) if owner_user_id else None,
        )
        return Chunk(chunk_id=str(chunk_id), text=text or "", metadata=meta,
                     vector_score=round(similarity, 4), score=round(similarity, 4),
                     similarity=round(similarity, 4), similarity_score=round(similarity, 4))

    def _ensure_schema(self) -> None:
        # This is idempotent and is also included in supabase/001_schema.sql for
        # operators who prefer to run migrations from the Supabase SQL editor.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS arise_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        owner_user_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        subject TEXT NOT NULL,
                        unit INTEGER,
                        page_number INTEGER NOT NULL,
                        char_start INTEGER NOT NULL DEFAULT 0,
                        char_end INTEGER NOT NULL DEFAULT 0,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        content TEXT NOT NULL,
                        embedding vector(384) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS arise_chunks_owner_idx ON arise_chunks(owner_user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS arise_chunks_doc_idx ON arise_chunks(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS arise_chunks_subject_idx ON arise_chunks(owner_user_id, subject)")
                cur.execute("CREATE INDEX IF NOT EXISTS arise_chunks_embedding_hnsw ON arise_chunks USING hnsw (embedding vector_cosine_ops)")
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM arise_chunks")
            return int(cur.fetchone()[0])

    def get_stored_subjects(self, owner_user_id: Optional[str] = None) -> List[str]:
        sql = "SELECT DISTINCT subject FROM arise_chunks"
        args: List[Any] = []
        if owner_user_id:
            sql += " WHERE owner_user_id = %s"
            args.append(owner_user_id)
        sql += " ORDER BY subject"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            subjects = {str(row[0]).strip() for row in cur.fetchall() if row[0]}
        self._cached_subjects = subjects
        return sorted(subjects)

    def resolve_subject(self, subject: Optional[str], owner_user_id: Optional[str] = None) -> Optional[str]:
        if not subject or str(subject).strip().lower() == "all":
            return None
        clean = str(subject).strip()
        for stored in self.get_stored_subjects(owner_user_id):
            if stored.lower() == clean.lower():
                return stored
        return clean

    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        self._ensure_schema()
        vectors = self._embed([c.text for c in chunks])
        with self._connect() as conn, conn.cursor() as cur:
            for chunk, vector in zip(chunks, vectors):
                meta = chunk.metadata
                cur.execute("""
                    INSERT INTO arise_chunks
                      (chunk_id, doc_id, owner_user_id, filename, subject, unit,
                       page_number, char_start, char_end, token_count, content, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      doc_id=EXCLUDED.doc_id, owner_user_id=EXCLUDED.owner_user_id,
                      filename=EXCLUDED.filename, subject=EXCLUDED.subject,
                      unit=EXCLUDED.unit, page_number=EXCLUDED.page_number,
                      char_start=EXCLUDED.char_start, char_end=EXCLUDED.char_end,
                      token_count=EXCLUDED.token_count, content=EXCLUDED.content,
                      embedding=EXCLUDED.embedding
                """, (chunk.chunk_id, meta.doc_id, meta.owner_user_id or "", meta.filename,
                      meta.subject, meta.unit, meta.page_number, meta.char_start,
                      meta.char_end, meta.token_count, chunk.text, self._vector_literal(vector)))
            conn.commit()
        self._cached_subjects = None
        return len(chunks)

    def search(self, query: str, subject: Optional[str] = None, unit: Optional[str] = None,
               top_k: int = 5, owner_user_id: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        self._ensure_schema()
        query_vector = self._vector_literal(self._embed([query])[0])
        conditions = ["owner_user_id = %s"]
        args: List[Any] = [owner_user_id or ""]
        resolved_subject = self.resolve_subject(subject, owner_user_id)
        if resolved_subject:
            conditions.append("subject = %s")
            args.append(resolved_subject)
        unit_value = self._unit_value(unit)
        if unit_value is not None:
            conditions.append("unit = %s")
            args.append(unit_value)
        args.extend([query_vector, max(1, min(int(top_k), 100))])
        sql = f"""
            SELECT chunk_id, content, doc_id, filename, subject, unit, page_number,
                   char_start, char_end, token_count, owner_user_id,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM arise_chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # The query vector is needed once for score and once for ordering.
        args = [query_vector] + args[:-2] + [query_vector, args[-1]]
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
        except Exception as exc:
            raise PgVectorSearchError(f"Supabase pgvector search failed: {exc}") from exc
        result = [(self._from_row(row), float(row[-1] or 0.0)) for row in rows]
        result.sort(key=lambda item: item[1], reverse=True)
        return result

    def get_all_chunks(self, owner_user_id: Optional[str] = None) -> List[Chunk]:
        sql = """SELECT chunk_id, content, doc_id, filename, subject, unit, page_number,
                         char_start, char_end, token_count, owner_user_id, 1.0
                  FROM arise_chunks"""
        args: List[Any] = []
        if owner_user_id:
            sql += " WHERE owner_user_id = %s"
            args.append(owner_user_id)
        sql += " ORDER BY created_at, chunk_id"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            return [self._from_row(row) for row in cur.fetchall()]

    def get_distinct_subjects_and_units(self, owner_user_id: Optional[str] = None) -> Dict[str, Any]:
        chunks = self.get_all_chunks(owner_user_id)
        subjects = sorted({c.metadata.subject for c in chunks if c.metadata.subject})
        units: Dict[str, set] = {s: set() for s in subjects}
        for chunk in chunks:
            if chunk.metadata.unit is not None:
                units.setdefault(chunk.metadata.subject, set()).add(int(chunk.metadata.unit))
        return {
            "subjects": subjects,
            "units_by_subject": {s: [f"Unit {n}" for n in sorted(vals)] for s, vals in units.items()},
            "total_documents": len({c.metadata.doc_id for c in chunks if c.metadata.doc_id}),
            "total_chunks": len(chunks),
        }

    def delete_document(self, doc_id: str) -> None:
        self._ensure_schema()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM arise_chunks WHERE doc_id = %s", (doc_id,))
            conn.commit()
        self._cached_subjects = None
