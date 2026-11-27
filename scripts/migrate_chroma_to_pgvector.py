"""Non-destructive Chroma -> Supabase pgvector migration.

Run with VECTOR_STORE_TYPE=pgvector and SUPABASE_DB_URL configured. The local
Chroma directory remains untouched. Run the Supabase SQL migration first.
"""
from __future__ import annotations
import os
import sys

# Keep app imports deterministic when run from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.rag.vector_store import _LegacyChromaVectorStore
from backend.app.rag.pgvector_store import SupabasePgVectorStore


def main() -> int:
    source = _LegacyChromaVectorStore()
    chunks = source.get_all_chunks()
    print(f"Found {len(chunks)} local Chroma chunks. Source will not be modified.")
    if not chunks:
        return 0
    target = SupabasePgVectorStore()
    batch_size = 32
    migrated = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        migrated += target.add_chunks(batch)
        print(f"Migrated {migrated}/{len(chunks)} chunks")
    print(f"Migration complete. Supabase pgvector now contains at least {target.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
