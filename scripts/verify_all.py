import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService
from backend.app.rag.llm import GeminiAnswerGenerator, NOT_FOUND_MESSAGE
from backend.app.rag.guardrail import evaluate

vs = ChromaVectorStore()
bm25 = BM25Index()
bm25.build_index(vs.get_all_chunks())
hybrid = HybridSearchService(vs, bm25)
llm = GeminiAnswerGenerator()

test_cases = [
    {
        "name": "Nielsen question, Hybrid, All subjects",
        "query": "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
        "subject": None,
        "mode": "hybrid",
        "expect_answer": True
    },
    {
        "name": "Nielsen question, Hybrid, Subject = Human Computer Interaction",
        "query": "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
        "subject": "Human Computer Interaction",
        "mode": "hybrid",
        "expect_answer": True
    },
    {
        "name": "Nielsen question, Vector, Subject = Human Computer Interaction",
        "query": "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
        "subject": "Human Computer Interaction",
        "mode": "vector",
        "expect_answer": True
    },
    {
        "name": "Topic-card prompt, Vector mode",
        "query": "Explain Nielsen's 10 Usability Heuristics in Human Computer Interaction",
        "subject": "Human Computer Interaction",
        "mode": "vector",
        "expect_answer": True
    },
    {
        "name": "Topic-card prompt, Hybrid mode",
        "query": "Explain Nielsen's 10 Usability Heuristics in Human Computer Interaction",
        "subject": "Human Computer Interaction",
        "mode": "hybrid",
        "expect_answer": True
    },
    {
        "name": "Off-topic 'Who won the IPL?', Vector mode",
        "query": "Who won the IPL?",
        "subject": None,
        "mode": "vector",
        "expect_answer": False
    },
    {
        "name": "Off-topic 'Who won the IPL?', Hybrid mode",
        "query": "Who won the IPL?",
        "subject": None,
        "mode": "hybrid",
        "expect_answer": False
    }
]

print("=" * 80)
print("             ARISE VERIFICATION TEST SUITE (End-to-End)")
print("=" * 80)

results_summary = []

for idx, tc in enumerate(test_cases, 1):
    print(f"\n[{idx}/7] Running: {tc['name']}")
    print(f"    Query:   \"{tc['query']}\"")
    print(f"    Subject: {tc['subject']} | Mode: {tc['mode']}")

    # 1. Search
    chunks = hybrid.search(
        query=tc['query'],
        subject=tc['subject'],
        unit=None,
        mode=tc['mode'],
        top_k=3
    )

    # 2. Guardrail
    g_res = evaluate(chunks=chunks, query=tc['query'], threshold=0.32, mode=tc['mode'])
    print(f"    Guardrail passed: {g_res.passed} | Reason: {g_res.reason}")

    # 3. Answer generation
    if not g_res.passed:
        ans = NOT_FOUND_MESSAGE
    else:
        ans = llm.generate_answer(
            query=tc['query'],
            chunks=chunks,
            threshold=0.32,
            mode=tc['mode']
        )

    is_fallback = (NOT_FOUND_MESSAGE.lower() in ans.lower())
    passed_test = (not is_fallback) if tc['expect_answer'] else is_fallback

    status_str = "PASS [OK]" if passed_test else "FAIL"
    print(f"    Answer preview: {ans[:140].strip()}...")
    print(f"    Test status: {status_str}")

    results_summary.append({
        "name": tc["name"],
        "mode": tc["mode"],
        "subject": str(tc["subject"]),
        "guardrail_passed": g_res.passed,
        "is_fallback": is_fallback,
        "test_status": status_str,
        "preview": ans[:100].replace("\n", " ")
    })

print("\n" + "=" * 80)
print("                      SUMMARY TABLE")
print("=" * 80)
for r in results_summary:
    print(f"{r['test_status']} | {r['name']:<55} | Fallback={r['is_fallback']}")

