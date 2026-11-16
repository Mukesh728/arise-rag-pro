import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.llm import GeminiAnswerGenerator

vs = ChromaVectorStore()
all_chunks = vs.get_all_chunks()

c1 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 1][0]
c10 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 10][0]
c11 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 11][0]
c13 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 13][0]

llm = GeminiAnswerGenerator()

# Test with [c1, c13, c10] (the top 3 chunks retrieved)
print("--- TEST WITH TOP 3 CHUNKS (p1, p13, p10) ---")
ans_top3 = llm.generate_answer(
    query="Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
    chunks=[c1, c13, c10],
    threshold=0.32,
    mode="hybrid"
)
print("Ans with p1, p13, p10:\n", ans_top3)

# Test with [c10, c11, c13] (includes p11 which has heuristics 9 & 10!)
print("\n--- TEST WITH CHUNKS INCLUDING PAGE 11 (p10, p11, c13) ---")
ans_p11 = llm.generate_answer(
    query="Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
    chunks=[c10, c11, c13],
    threshold=0.32,
    mode="hybrid"
)
print("Ans with p10, p11, p13:\n", ans_p11)

