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

q = "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation"

print("=== ALL SUBJECTS (Test 1) ===")
for idx, c in enumerate(hybrid.search(query=q, subject=None, mode='hybrid', top_k=3), 1):
    print(f"Chunk {idx}: doc={c.metadata.doc_id} subj='{c.metadata.subject}' page={c.metadata.page_number}")
    print(f"Text snippet: {c.text[:200]}...\n")

print("=== SUBJECT: Human Computer Interaction (Test 2) ===")
for idx, c in enumerate(hybrid.search(query=q, subject='Human Computer Interaction', mode='hybrid', top_k=3), 1):
    print(f"Chunk {idx}: doc={c.metadata.doc_id} subj='{c.metadata.subject}' page={c.metadata.page_number}")
    print(f"Text snippet: {c.text[:200]}...\n")
