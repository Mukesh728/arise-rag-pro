import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.config import settings
from backend.app.models import QueryRequest
from backend.app.rag.llm import GeminiAnswerGenerator, NOT_FOUND_MESSAGE, LLM_UNAVAILABLE_MESSAGE
from backend.app.main import query_rag, expand_neighbor_chunks, vector_store, hybrid_service
from unittest.mock import patch, MagicMock

def test_phase4():
    print("--- 1. Testing Default Top-K = 5 ---")
    assert settings.DEFAULT_TOP_K == 5, f"Expected 5, got {settings.DEFAULT_TOP_K}"
    qr = QueryRequest(query="test")
    assert qr.top_k == 5, f"Expected QueryRequest default 5, got {qr.top_k}"
    print("Default Top-K = 5 PASS")

    print("\n--- 2. Testing Configurable Model ---")
    assert settings.GEMINI_MODEL in ("gemini-3.5-flash", "gemini-3.5-flash-lite")
    gen = GeminiAnswerGenerator(api_key="mock")
    assert gen.model == settings.GEMINI_MODEL
    print("Configurable model PASS")

    print("\n--- 3. Testing Accurate Fallback Messages ---")
    # Test IPL query (Guardrail fails)
    req_ipl = QueryRequest(query="Who won the IPL?", mode="hybrid", subject="Human Computer Interaction")
    res_ipl = query_rag(req_ipl)
    assert res_ipl.is_fallback == True
    assert "Retrieval relevance too low" in res_ipl.fallback_message
    print("IPL low relevance fallback message PASS:", res_ipl.fallback_message)

    # Test query where guardrail passes but LLM returns NOT_FOUND_MESSAGE
    req_nielsen = QueryRequest(
        query="Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
        mode="hybrid",
        subject="Human Computer Interaction"
    )
    with patch("backend.app.main.llm_generator.generate_answer", return_value=NOT_FOUND_MESSAGE):
        res_mock_nf = query_rag(req_nielsen)
        assert res_mock_nf.is_fallback == True
        assert "do not contain enough information" in res_mock_nf.fallback_message
        print("Guardrail passed but LLM not found message PASS:", res_mock_nf.fallback_message)

    print("\n--- 4. Testing LLM Failure Handling ---")
    with patch("backend.app.main.llm_generator.generate_answer", side_effect=Exception("429 Resource Exhausted")):
        res_llm_fail = query_rag(req_nielsen)
        assert res_llm_fail.answer == LLM_UNAVAILABLE_MESSAGE
        assert not res_llm_fail.is_fallback
        assert len(res_llm_fail.citations) > 0
        print("LLM exception handled gracefully with citations PASS:", res_llm_fail.answer)

    print("\n--- 5. Testing Nielsen Retrieval Baseline ---")
    # Vector mode
    req_v = QueryRequest(
        query="Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
        mode="vector",
        subject="Human Computer Interaction",
        top_k=5
    )
    with patch("backend.app.main.llm_generator.generate_answer", return_value="[Mocked Grounded Answer citing Page 10, 11, 12]"):
        res_v = query_rag(req_v)
        pages_v = [c.page_number for c in res_v.citations]
        print(f"Vector retrieved {len(res_v.citations)} citations on pages: {pages_v}")
        assert not res_v.is_fallback

    # Hybrid mode
    req_h = QueryRequest(
        query="Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
        mode="hybrid",
        subject="Human Computer Interaction",
        top_k=5
    )
    with patch("backend.app.main.llm_generator.generate_answer", return_value="[Mocked Grounded Answer citing Page 10, 11, 12]"):
        res_h = query_rag(req_h)
        pages_h = [c.page_number for c in res_h.citations]
        print(f"Hybrid retrieved {len(res_h.citations)} citations on pages: {pages_h}")
        assert not res_h.is_fallback

    # Test neighbor chunk expansion
    all_chunks = vector_store.get_all_chunks()
    top_chunks = hybrid_service.search(req_h.query, subject="Human Computer Interaction", mode="hybrid", top_k=5)
    expanded = expand_neighbor_chunks(top_chunks, all_chunks, cap=8)
    exp_pages = [c.metadata.page_number for c in expanded]
    print(f"Neighbor-expanded context pages (cap 8): {exp_pages}")
    assert len(expanded) >= len(top_chunks)

    print("\nALL PHASE 4 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_phase4()
