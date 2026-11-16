import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService
from backend.app.rag.llm import GeminiAnswerGenerator

vs = ChromaVectorStore()
bm25 = BM25Index()
bm25.build_index(vs.get_all_chunks())
hybrid = HybridSearchService(vs, bm25)
llm = GeminiAnswerGenerator()

q = "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation"
chunks = hybrid.search(query=q, subject="Human Computer Interaction", mode="hybrid", top_k=3)

print("Retrieved chunks:")
for c in chunks:
    print(f"  page={c.metadata.page_number} doc={c.metadata.doc_id}")

ans = llm.generate_answer(query=q, chunks=chunks, threshold=0.32, mode="hybrid")
print("\nGEMINI RESPONSE:")
print(ans)

