"""
Fast offline benchmark: embed each query ONCE at K=10, then slice results at K=1,2,3,5.
Avoids re-embedding (which is the slow step) for each K value.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService

# 20 benchmark queries from frontend/app.js defaultBenchmarkQueries
BENCHMARK_QUERIES = [
    {"id": 1,  "question": "What are the seven graphical design concepts including Balance and visual weight in screen design?", "expected_page": 3,  "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 2,  "question": "What is visible language and what are its primary components such as typography and layout?", "expected_page": 4,  "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 3,  "question": "What is grid-based design and what role do columns, rows, gutters and margins play?", "expected_page": 6,  "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 4,  "question": "Explain usability design principles and concepts derived from Don Norman such as affordance and feedback.", "expected_page": 7,  "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 5,  "question": "What are Ben Shneiderman's Eight Golden Rules for interface design?", "expected_page": 8,  "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 6,  "question": "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation.", "expected_page": 10, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 7,  "question": "Compare Shneiderman's Golden Rules with Nielsen's Heuristics in terms of design guidelines vs evaluation checklist.", "expected_page": 12, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 8,  "question": "What is Heuristic Evaluation and why are 3 to 5 expert evaluators recommended?", "expected_page": 13, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 9,  "question": "What is an HCI design pattern and when should designers use reusable patterns?", "expected_page": 14, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 10, "question": "How should a student structure answers in an HCI viva including definition, purpose and example?", "expected_page": 15, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 11, "question": "What are the core subjects and course codes listed in MRSPTU 7th Semester B.Tech CSE study scheme?", "expected_page": 2,  "subject": "CSE Sem 7 Syllabus", "unit": None},
    {"id": 12, "question": "What are the course objectives, outcomes, and contents of Soft Computing BCSED1-712?", "expected_page": 3,  "subject": "CSE Sem 7 Syllabus", "unit": None},
    {"id": 13, "question": "What are the syllabus topics and recommended textbooks by Dix and Finlay for Human Computer Interaction BCSED1-713?", "expected_page": 4,  "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 14, "question": "What are the course outcomes and syllabus topics for Cryptography & Network Security BCSED1-723?", "expected_page": 5,  "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 15, "question": "What topics are covered in Artificial Intelligence BCSED1-724 including heuristic search and knowledge representation?", "expected_page": 6,  "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 16, "question": "What natural resources and forest or water conservation topics are covered in Environmental Sciences Unit-I?", "expected_page": 7,  "subject": "CSE Sem 7 Syllabus", "unit": "Unit 1"},
    {"id": 17, "question": "What environmental pollution types and human health topics are covered in Environmental Sciences?", "expected_page": 8,  "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 18, "question": "Explain the concepts of ecosystems, producers, consumers, and food chains in the syllabus.", "expected_page": 9,  "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 19, "question": "What topics are covered in Constitution of India BMNCC0-001 including Directive Principles and Article 19 and 21?", "expected_page": 10, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 20, "question": "What is the difference between Recognition vs Recall and Affordance vs Mapping in UI design?", "expected_page": 12, "subject": "Human Computer Interaction", "unit": "Unit 3"},
]

MAX_K = 10  # retrieve at max K once, slice at smaller K values
K_VALS = [1, 2, 3, 5]

def run():
    print("Initialising services...")
    vector_store = ChromaVectorStore()
    bm25_index = BM25Index()
    all_chunks = vector_store.get_all_chunks()
    bm25_index.build_index(all_chunks)
    svc = HybridSearchService(vector_store=vector_store, bm25_index=bm25_index)
    print(f"Loaded {len(all_chunks)} chunks. Embedding {len(BENCHMARK_QUERIES)} queries once at K={MAX_K}...")

    # Cache results at MAX_K for each mode
    results = {}
    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        subj = q["subject"]
        unit = q["unit"]
        question = q["question"]
        expected = q["expected_page"]
        print(f"  [{i:02d}/{len(BENCHMARK_QUERIES)}] {question[:60]}...")
        v_chunks = svc.search(query=question, subject=subj, unit=unit, mode="vector", top_k=MAX_K)
        h_chunks = svc.search(query=question, subject=subj, unit=unit, mode="hybrid", top_k=MAX_K)
        results[q["id"]] = {
            "expected": expected,
            "v_pages": [c.metadata.page_number for c in v_chunks],
            "h_pages": [c.metadata.page_number for c in h_chunks],
        }

    print("\n--- Hit Rate@K Results ---")
    print(f"{'K':<4} {'Vector Hits':<14} {'Vector HR%':<13} {'Hybrid Hits':<14} {'Hybrid HR%':<13} {'Delta'}")
    print("-" * 70)
    total = len(BENCHMARK_QUERIES)
    for k in K_VALS:
        v_hits = sum(1 for r in results.values() if r["expected"] in r["v_pages"][:k])
        h_hits = sum(1 for r in results.values() if r["expected"] in r["h_pages"][:k])
        v_hr = v_hits / total * 100
        h_hr = h_hits / total * 100
        delta = h_hr - v_hr
        print(f"{k:<4} {v_hits}/{total:<11} {v_hr:.1f}%{'':<9} {h_hits}/{total:<11} {h_hr:.1f}%{'':<9} {delta:+.1f}%")

    print("\n--- Per-query detail at K=5 ---")
    print(f"{'ID':<4} {'Expected Pg':<13} {'Vector Pgs':<25} {'Hybrid Pgs':<25} {'V-Hit':<7} {'H-Hit'}")
    print("-" * 90)
    for q in BENCHMARK_QUERIES:
        r = results[q["id"]]
        vp = str(r["v_pages"][:5])
        hp = str(r["h_pages"][:5])
        vh = "✓" if r["expected"] in r["v_pages"][:5] else "✗"
        hh = "✓" if r["expected"] in r["h_pages"][:5] else "✗"
        print(f"{q['id']:<4} {r['expected']:<13} {vp:<25} {hp:<25} {vh:<7} {hh}")

if __name__ == "__main__":
    run()
