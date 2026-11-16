"""
Full system check — runs every major function and prints PASS/FAIL.
Run: python scripts/full_check.py
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService
from backend.app.rag.guardrail import evaluate
from backend.app.rag.llm import GeminiAnswerGenerator, MODEL_NAME, NOT_FOUND_MESSAGE
from backend.app.config import settings

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, ok, detail=""):
    icon = PASS if ok else FAIL
    print(f"  [{icon}] {name}" + (f"  — {detail}" if detail else ""))
    results.append((name, ok))

print("\n" + "="*60)
print("  ARISE RAG — Full System Check")
print("="*60)

# ── 1. Config ──────────────────────────────────────────────
print("\n[1] Config")
check("API key present", bool(settings.GEMINI_API_KEY), f"key starts with {settings.GEMINI_API_KEY[:6]}...")
check("Model name in llm.py", MODEL_NAME == "gemini-3.8-flash", f"MODEL_NAME = '{MODEL_NAME}'")
check("Threshold", settings.SIMILARITY_THRESHOLD == 0.32, f"{settings.SIMILARITY_THRESHOLD}")

# ── 2. ChromaDB ────────────────────────────────────────────
print("\n[2] ChromaDB / Vector Store")
try:
    vs = ChromaVectorStore()
    total = vs.count()
    check("ChromaDB connects", True, f"{total} chunks stored")
    check("Has chunks", total > 0, f"total={total}")

    subjects = vs.get_stored_subjects()
    check("get_stored_subjects()", len(subjects) > 0, str(subjects))

    resolved = vs.resolve_subject("human computer interaction")
    check("resolve_subject() case-insensitive", resolved == "Human Computer Interaction",
          f"'human computer interaction' → '{resolved}'")

    vec_results = vs.search("Nielsen usability heuristics", subject=None, top_k=3)
    check("Vector search (no filter)", len(vec_results) > 0, f"{len(vec_results)} results")

    vec_hci = vs.search("Nielsen usability heuristics",
                        subject="Human Computer Interaction", top_k=3)
    check("Vector search (HCI filter)", len(vec_hci) > 0, f"{len(vec_hci)} results, top score={vec_hci[0][1]:.4f}" if vec_hci else "0 results")
except Exception as e:
    check("ChromaDB init", False, str(e))

# ── 3. BM25 ────────────────────────────────────────────────
print("\n[3] BM25 Index")
try:
    bm25 = BM25Index()
    all_chunks = vs.get_all_chunks()
    bm25.build_index(all_chunks)
    check("BM25 index built", True, f"{len(all_chunks)} chunks indexed")

    bm25_results = bm25.search("Nielsen heuristics usability", top_k=3)
    check("BM25 search (no filter)", len(bm25_results) > 0, f"{len(bm25_results)} results")

    bm25_hci = bm25.search("Nielsen heuristics", subject="Human Computer Interaction", top_k=3)
    check("BM25 search (HCI filter)", len(bm25_hci) > 0, f"{len(bm25_hci)} results")
except Exception as e:
    check("BM25 init", False, str(e))

# ── 4. Hybrid Search ───────────────────────────────────────
print("\n[4] Hybrid Search (RRF)")
try:
    svc = HybridSearchService(vector_store=vs, bm25_index=bm25)

    # Test hybrid mode
    hyb = svc.search("Explain Nielsen 10 usability heuristics", subject=None, mode="hybrid", top_k=3)
    check("Hybrid search all subjects", len(hyb) > 0, f"{len(hyb)} chunks, top vector_score={hyb[0].vector_score}")

    hyb_hci = svc.search("Explain Nielsen 10 usability heuristics",
                         subject="human computer interaction", mode="hybrid", top_k=3)
    check("Hybrid search HCI (lowercase input)", len(hyb_hci) > 0,
          f"{len(hyb_hci)} chunks, subject resolved correctly")

    # Test vector mode
    vec = svc.search("Nielsen 10 usability heuristics", subject="human computer interaction",
                     mode="vector", top_k=3)
    check("Vector mode HCI (lowercase input)", len(vec) > 0,
          f"{len(vec)} chunks, vector_score={vec[0].vector_score:.4f}" if vec else "0 chunks")
except Exception as e:
    check("Hybrid search", False, str(e))

# ── 5. Guardrail ───────────────────────────────────────────
print("\n[5] Guardrail")
try:
    # On-topic hybrid query
    g1 = evaluate(hyb_hci, "Nielsen usability heuristics", threshold=0.32, mode="hybrid")
    check("Guardrail PASSES for HCI chunks (hybrid)", g1.passed,
          f"max_sim={g1.max_dense_similarity:.4f} | {g1.reason[:80]}")

    # On-topic vector query
    g2 = evaluate(vec, "Nielsen usability heuristics", threshold=0.32, mode="vector")
    check("Guardrail PASSES for HCI chunks (vector)", g2.passed,
          f"max_sim={g2.max_dense_similarity:.4f} | {g2.reason[:80]}")

    # Off-topic
    off_topic_chunks = svc.search("Who won the IPL cricket tournament", subject=None, mode="hybrid", top_k=3)
    g3 = evaluate(off_topic_chunks, "Who won the IPL cricket tournament", threshold=0.32, mode="hybrid")
    check("Guardrail REJECTS off-topic 'IPL'", not g3.passed,
          f"max_sim={g3.max_dense_similarity:.4f}")
except Exception as e:
    check("Guardrail", False, str(e))

# ── 6. Gemini LLM ─────────────────────────────────────────
print("\n[6] Gemini LLM (requires network)")
try:
    llm = GeminiAnswerGenerator()
    check("Gemini client initialized", llm.client is not None, f"model={MODEL_NAME}")

    # Answer an HCI question
    answer = llm.generate_answer(
        query="What are Nielsen's 10 usability heuristics?",
        chunks=hyb_hci,
        threshold=0.32,
        mode="hybrid"
    )
    is_fallback = NOT_FOUND_MESSAGE.lower() in answer.lower()
    check("Gemini returns REAL answer (hybrid+HCI)", not is_fallback,
          answer[:120].replace('\n', ' ') + "..." if not is_fallback else "FALLBACK returned")

    # Off-topic should get fallback
    answer_off = llm.generate_answer(
        query="Who won the IPL cricket tournament?",
        chunks=off_topic_chunks,
        threshold=0.32,
        mode="hybrid"
    )
    is_fb_off = NOT_FOUND_MESSAGE.lower() in answer_off.lower()
    check("Gemini returns FALLBACK for off-topic", is_fb_off,
          "correctly returned not-found")
except Exception as e:
    check("Gemini LLM", False, str(e))

# ── Summary ────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
total_tests = len(results)
failed = [(n, ok) for n, ok in results if not ok]
print(f"  Result: {passed}/{total_tests} checks passed")
if failed:
    print("\n  FAILED checks:")
    for n, _ in failed:
        print(f"    ✗ {n}")
print("="*60 + "\n")

