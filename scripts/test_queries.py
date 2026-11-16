import os
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

queries = [
    ("Explain Nielsen's 10 Usability Heuristics in Human Computer Interaction", "Topic Card Query"),
    ("Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation.", "Viva Tab Query"),
    ("Who won the IPL?", "Off-topic Query")
]

for q, qname in queries:
    print(f"\n==========================================")
    print(f"QUERY: {qname}: '{q}'")
    print(f"==========================================")
    for subj in [None, "Human Computer Interaction"]:
        subj_label = subj if subj else "All Subjects"
        for mode in ["vector", "hybrid"]:
            chunks = hybrid.search(query=q, subject=subj, unit=None, mode=mode, top_k=3)
            print(f"\n[Mode={mode.upper()} | Subject={subj_label}] -> Returned {len(chunks)} chunks")
            for idx, c in enumerate(chunks, 1):
                print(f"  {idx}. doc={c.metadata.doc_id} | subj='{c.metadata.subject}' | p={c.metadata.page_number} | v_score={c.vector_score} | bm25={c.bm25_score} | rrf={c.rrf_score}")
                print(f"     score attr={c.score} | similarity attr={c.similarity}")
            
            # Now let's see how main.py / llm.py extracts score:
            valid_vector_scores = [
                getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', getattr(c, 'similarity_score', None))))
                for c in chunks
            ]
            valid_vector_scores = [s for s in valid_vector_scores if s is not None]
            max_vector_score = max(valid_vector_scores) if valid_vector_scores else 0.0
            print(f"  -> Guardrail check: max_vector_score={max_vector_score} (threshold=0.32) => Pass: {max_vector_score >= 0.32}")

