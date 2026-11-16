import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService
# Evaluation baseline test

# The 20 benchmark queries from app.js
default_queries = [
    {"id": 1, "question": "What are the seven graphical design concepts including Balance and visual weight in screen design?", "expected_page": 3, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 2, "question": "Explain Equilibrium, Symmetry, and Proportion in screen presentation guidelines.", "expected_page": 4, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 3, "question": "What is Visual Rhythm and Regularity in GUI screen layout?", "expected_page": 5, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 4, "question": "Explain Predictability, Economy, and Sequentiality in screen structure.", "expected_page": 6, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 5, "question": "What is Alignment, Grouping, and Visual Clarity in user interface design?", "expected_page": 7, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 6, "question": "What are the common screen design pitfalls according to Galitz?", "expected_page": 8, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 7, "question": "Explain Ben Shneiderman's 8 Golden Rules of Interface Design in detail.", "expected_page": 9, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 8, "question": "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation.", "expected_page": 10, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 9, "question": "Compare Shneiderman's Golden Rules with Nielsen's Heuristics in terms of design guidelines vs evaluation checklist.", "expected_page": 12, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 10, "question": "What is Heuristic Evaluation and how is the 1 to 4 severity rating scale applied?", "expected_page": 15, "subject": "Human Computer Interaction", "unit": "Unit 3"},
    {"id": 11, "question": "What are the core subjects and course codes listed in MRSPTU 7th Semester B.Tech CSE study scheme?", "expected_page": 2, "subject": "CSE Sem 7 Syllabus", "unit": None},
    {"id": 12, "question": "What are the course objectives, outcomes, and contents of Soft Computing BCSED1-712?", "expected_page": 3, "subject": "CSE Sem 7 Syllabus", "unit": None},
    {"id": 13, "question": "What are the syllabus topics and recommended textbooks by Dix and Finlay for Human Computer Interaction BCSED1-713?", "expected_page": 4, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 14, "question": "What are the course outcomes and syllabus topics for Cryptography & Network Security BCSED1-723?", "expected_page": 5, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 15, "question": "What topics are covered in Artificial Intelligence BCSED1-724 including heuristic search and knowledge representation?", "expected_page": 6, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 16, "question": "What natural resources and forest or water conservation topics are covered in Environmental Sciences Unit-I?", "expected_page": 7, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 1"},
    {"id": 17, "question": "What environmental pollution types and human health topics are covered in Environmental Sciences?", "expected_page": 8, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 18, "question": "Explain the concepts of ecosystems, producers, consumers, and food chains in the syllabus.", "expected_page": 9, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 19, "question": "What topics are covered in Constitution of India BMNCC0-001 including Directive Principles and Article 19 and 21?", "expected_page": 10, "subject": "CSE Sem 7 Syllabus", "unit": "Unit 4"},
    {"id": 20, "question": "What is the difference between Recognition vs Recall and Affordance vs Mapping in UI design?", "expected_page": 12, "subject": "Human Computer Interaction", "unit": "Unit 3"}
]

vs = ChromaVectorStore()
bm25 = BM25Index()
bm25.build_index(vs.get_all_chunks())
hybrid = HybridSearchService(vs, bm25)

v_hits = 0
h_hits = 0
top_k = 3

for q in default_queries:
    subj = q["subject"]
    unit = q["unit"]
    exp_p = q["expected_page"]

    v_chunks = hybrid.search(query=q["question"], subject=subj, unit=unit, mode="vector", top_k=top_k)
    v_pages = [c.metadata.page_number for c in v_chunks]
    v_hit = exp_p in v_pages
    if v_hit:
        v_hits += 1

    h_chunks = hybrid.search(query=q["question"], subject=subj, unit=unit, mode="hybrid", top_k=top_k)
    h_pages = [c.metadata.page_number for c in h_chunks]
    h_hit = exp_p in h_pages
    if h_hit:
        h_hits += 1

total = len(default_queries)
print(f"BASELINE EVALUATION (top_k={top_k}):")
print(f"  Vector Hits: {v_hits}/{total} ({v_hits/total*100:.1f}%)")
print(f"  Hybrid Hits: {h_hits}/{total} ({h_hits/total*100:.1f}%)")
print(f"  Delta: +{(h_hits - v_hits)/total*100:.1f}%")
