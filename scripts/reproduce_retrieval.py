import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService

def test_reproduction():
    print("=== STEP 1: INITIALIZING STORES ===")
    vs = ChromaVectorStore()
    total_chunks = vs.count()
    print(f"ChromaDB chunk count: {total_chunks}")

    bm25 = BM25Index()
    all_chunks = vs.get_all_chunks()
    bm25.build_index(all_chunks)
    print(f"BM25 index built with {len(all_chunks)} chunks.")

    hybrid = HybridSearchService(vs, bm25)

    # Let's inspect stored subjects and doc_ids
    subjects_info = vs.get_distinct_subjects_and_units()
    print("\n=== DISTINCT SUBJECTS & DOC STATS ===")
    print("Subjects in Chroma:", subjects_info["subjects"])
    print("Units by subject:", subjects_info["units_by_subject"])

    # Inspect doc_id and subject distribution across all chunks
    docs_summary = {}
    for c in all_chunks:
        d_id = c.metadata.doc_id
        subj = c.metadata.subject
        fname = c.metadata.filename
        key = (d_id, subj, fname)
        docs_summary[key] = docs_summary.get(key, 0) + 1

    print("\nDocuments summary (doc_id, subject, filename -> chunk_count):")
    for (d_id, subj, fname), count in docs_summary.items():
        print(f"  - doc_id: {d_id} | subject: '{subj}' | filename: '{fname}' | chunks: {count}")

    # Test Vector Search raw query to check embedding function health (Root cause B)
    print("\n=== TESTING VECTOR SEARCH HEALTH (Root cause B) ===")
    test_query = "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation"
    try:
        raw_vector_results = vs.collection.query(
            query_texts=[test_query],
            n_results=min(5, total_chunks)
        )
        print("Raw Chroma query succeeded!")
        print(f"Returned {len(raw_vector_results.get('ids', [[]])[0])} results.")
        if raw_vector_results.get('distances'):
            print(f"Distances: {raw_vector_results['distances'][0]}")
            sims = [1.0 - d for d in raw_vector_results['distances'][0]]
            print(f"Derived similarities: {sims}")
    except Exception as e:
        import traceback
        print("Raw Chroma query FAILED with exception:")
        traceback.print_exc()

    # Now test the 4 combinations requested:
    # Mode: [vector, hybrid] x Subject: [None ("all"), "Human Computer Interaction"]
    cases = [
        ("vector", None, "Vector | All Subjects"),
        ("vector", "Human Computer Interaction", "Vector | Subject='Human Computer Interaction'"),
        ("hybrid", None, "Hybrid | All Subjects"),
        ("hybrid", "Human Computer Interaction", "Hybrid | Subject='Human Computer Interaction'"),
    ]

    print("\n=== RUNNING 4 RETRIEVAL TEST COMBINATIONS ===")
    for mode, subj, label in cases:
        print(f"\n--- Test Case: {label} ---")
        try:
            chunks = hybrid.search(
                query=test_query,
                subject=subj,
                unit=None,
                mode=mode,
                top_k=3
            )
            print(f"Retrieved {len(chunks)} chunks.")
            for idx, c in enumerate(chunks, 1):
                # Check nested getattr logic as currently implemented in main.py / llm.py
                nested_getattr = getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', getattr(c, 'similarity_score', None))))
                print(f"  Chunk {idx}:")
                print(f"    chunk_id: {c.chunk_id} | doc_id: {c.metadata.doc_id} | subject: '{c.metadata.subject}' | page: {c.metadata.page_number}")
                print(f"    vector_score: {c.vector_score}")
                print(f"    bm25_score: {c.bm25_score}")
                print(f"    rrf_score: {c.rrf_score}")
                print(f"    nested_getattr result: {nested_getattr}")
                print(f"    c.score: {c.score}")
                print(f"    Snippet: {c.text[:120].strip()}...")

            # Current guardrail decision in main.py
            valid_vector_scores = [
                getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', getattr(c, 'similarity_score', None))))
                for c in chunks
            ]
            valid_vector_scores = [s for s in valid_vector_scores if s is not None]
            max_vector_score = max(valid_vector_scores) if valid_vector_scores else 0.0
            passed = bool(chunks and max_vector_score >= 0.32)
            print(f"  Current guardrail: valid_vector_scores={valid_vector_scores}, max={max_vector_score:.4f}, threshold=0.32 => PASSED: {passed}")

        except Exception as e:
            import traceback
            print(f"  ERROR in {label}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_reproduction()

