import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
from backend.app.main import query_rag
from backend.app.rag.llm import GeminiAnswerGenerator, NOT_FOUND_MESSAGE
from backend.app.rag.guardrail import evaluate
from backend.app.rag.hybrid_search import HybridSearchService
from backend.app.rag.vector_store import ChromaVectorStore, VectorSearchError
from backend.app.rag.bm25_search import BM25Index

def test_phase1():
    assert not inspect.iscoroutinefunction(query_rag), "query_rag should be synchronous def"
    print("query_rag is synchronous def: PASS")

    gen = GeminiAnswerGenerator(api_key="mock_key")
    print("GeminiAnswerGenerator instantiated successfully: PASS")

    vs = ChromaVectorStore()
    bm = BM25Index()
    bm.build_index(vs.get_all_chunks())
    hs = HybridSearchService(vs, bm)

    # Nielsen query
    q_nielsen = "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation"
    chunks_v = hs.search(q_nielsen, subject="Human Computer Interaction", mode="vector", top_k=3)
    res_v = evaluate(chunks_v, q_nielsen, threshold=0.32, mode="vector")
    print("Nielsen Vector Guardrail:", res_v.passed, "|", res_v.reason)
    assert res_v.passed, "Nielsen vector should pass"

    chunks_h = hs.search(q_nielsen, subject="Human Computer Interaction", mode="hybrid", top_k=3)
    res_h = evaluate(chunks_h, q_nielsen, threshold=0.32, mode="hybrid")
    print("Nielsen Hybrid Guardrail:", res_h.passed, "|", res_h.reason)
    assert res_h.passed, "Nielsen hybrid should pass"

    # IPL query
    q_ipl = "Who won the IPL?"
    chunks_ipl_v = hs.search(q_ipl, subject="Human Computer Interaction", mode="vector", top_k=3)
    res_ipl_v = evaluate(chunks_ipl_v, q_ipl, threshold=0.32, mode="vector")
    print("IPL Vector Guardrail:", res_ipl_v.passed, "|", res_ipl_v.reason)
    assert not res_ipl_v.passed, "IPL vector must be rejected"

    chunks_ipl_h = hs.search(q_ipl, subject="Human Computer Interaction", mode="hybrid", top_k=3)
    res_ipl_h = evaluate(chunks_ipl_h, q_ipl, threshold=0.32, mode="hybrid")
    print("IPL Hybrid Guardrail:", res_ipl_h.passed, "|", res_ipl_h.reason)
    assert not res_ipl_h.passed, "IPL hybrid must be rejected"

    print("\nALL PHASE 1 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_phase1()
