import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService

vs = ChromaVectorStore()
bm25 = BM25Index()
bm25.build_index(vs.get_all_chunks())
hybrid = HybridSearchService(vs, bm25)

# Test query with lower case subject
q = "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation."
subj = "human computer interaction" # lower case!

print(f"Querying with subject='{subj}'...")
chunks = hybrid.search(query=q, subject=subj, unit=None, mode="hybrid", top_k=3)
print(f"Retrieved {len(chunks)} chunks.")
for idx, c in enumerate(chunks, 1):
    print(f"  {idx}. doc={c.metadata.doc_id} | subject='{c.metadata.subject}' | page={c.metadata.page_number}")
    print(f"     v_score={c.vector_score} | bm25={c.bm25_score} | score={c.score} | similarity={c.similarity}")

# Old guardrail logic
valid_scores = [getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', getattr(c, 'similarity_score', None)))) for c in chunks]
valid_scores = [s for s in valid_scores if s is not None]
top_sim = max(valid_scores) if valid_scores else 0.0
print(f"Old Guardrail: valid_scores={valid_scores}, top_sim={top_sim} -> Pass: {top_sim >= 0.32}")

