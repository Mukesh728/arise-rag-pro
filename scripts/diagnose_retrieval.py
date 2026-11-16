"""
Diagnostic Tool for RAG Retrieval & ChromaDB Hygiene
=====================================================
Usage:
    python scripts/diagnose_retrieval.py [QUERY] [--subject SUBJECT] [--unit UNIT] [--threshold THRESHOLD] [--top-k K]

Features:
    (a) Total chunk count
    (b) Subjects with chunk counts and filenames, flagging UUID-like subjects
    (c) Duplicate chunk texts across different doc_ids
    (d) Embedding health check (raw vector search query)
    (e) Vector vs Hybrid retrieval results with scores and guardrail verdict
    * Strictly NO Gemini LLM calls are made in this script.
"""

import sys
import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore, VectorSearchError
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.hybrid_search import HybridSearchService
from backend.app.rag.guardrail import evaluate


UUID_PATTERN = re.compile(r'[0-9a-fA-F]{8}[-\s][0-9a-fA-F]{4}[-\s][0-9a-fA-F]{4}', re.IGNORECASE)


def is_uuid_like(text: str) -> bool:
    """Check if a string resembles a UUID."""
    if not text:
        return False
    return bool(UUID_PATTERN.search(text.strip()))


def diagnose(query: str, subject: str = None, unit: str = None, threshold: float = 0.32, top_k: int = 3):
    print("=" * 80)
    print("           ARISE RAG RETRIEVAL & CHROMADB DIAGNOSTIC REPORT")
    print("=" * 80)

    # Initialize Vector Store and BM25
    vs = ChromaVectorStore()
    total_chunks = vs.count()
    all_chunks = vs.get_all_chunks()

    bm25 = BM25Index()
    bm25.build_index(all_chunks)
    hybrid = HybridSearchService(vs, bm25)

    # -------------------------------------------------------------------------
    # (a) Total chunk count
    # -------------------------------------------------------------------------
    print(f"\n[SECTION A] TOTAL CHUNK COUNT")
    print(f"  Total chunks stored in ChromaDB: {total_chunks}")
    print(f"  Total chunks indexed in BM25:   {len(all_chunks)}")

    # -------------------------------------------------------------------------
    # (b) Subjects with chunk counts, filenames, and UUID-flagging
    # -------------------------------------------------------------------------
    print(f"\n[SECTION B] SUBJECTS, FILENAMES & DOCUMENT BREAKDOWN")
    
    # Map: subject -> { doc_id: { 'filename': fname, 'chunk_count': count, 'pages': set() } }
    subject_map: Dict[str, Dict[str, Dict]] = {}
    for c in all_chunks:
        subj = c.metadata.subject or "UNKNOWN_SUBJECT"
        d_id = c.metadata.doc_id or "UNKNOWN_DOC_ID"
        fname = c.metadata.filename or "UNKNOWN_FILENAME"
        p_num = c.metadata.page_number

        if subj not in subject_map:
            subject_map[subj] = {}
        if d_id not in subject_map[subj]:
            subject_map[subj][d_id] = {
                "filename": fname,
                "chunk_count": 0,
                "pages": set(),
            }
        subject_map[subj][d_id]["chunk_count"] += 1
        subject_map[subj][d_id]["pages"].add(p_num)

    uuid_flagged_count = 0
    for subj, docs in sorted(subject_map.items()):
        uuid_flag = " [!] WARNING: UUID-LIKE SUBJECT" if is_uuid_like(subj) else ""
        if uuid_flag:
            uuid_flagged_count += 1
        total_subj_chunks = sum(d["chunk_count"] for d in docs.values())
        print(f"\n  Subject: '{subj}' (Chunks: {total_subj_chunks}, Documents: {len(docs)}){uuid_flag}")
        for d_id, d_info in docs.items():
            fname = d_info["filename"]
            fname_uuid_flag = " [UUID filename]" if is_uuid_like(fname) else ""
            pages_span = f"pages {min(d_info['pages'])}-{max(d_info['pages'])}" if d_info['pages'] else "no pages"
            print(f"    - doc_id: {d_id} | file: '{fname}'{fname_uuid_flag} | {d_info['chunk_count']} chunks ({pages_span})")

    if uuid_flagged_count > 0:
        print(f"\n  * Total subjects flagged with UUID-like names: {uuid_flagged_count}")

    # -------------------------------------------------------------------------
    # (c) Duplicate chunk texts across different doc_ids
    # -------------------------------------------------------------------------
    print(f"\n[SECTION C] CROSS-DOCUMENT DUPLICATE CHUNK DETECTION")
    text_hashes: Dict[str, List] = {}
    for c in all_chunks:
        h = hashlib.sha256(c.text.strip().encode("utf-8")).hexdigest()
        if h not in text_hashes:
            text_hashes[h] = []
        text_hashes[h].append(c)

    duplicate_groups = {h: chunks for h, chunks in text_hashes.items() if len(set(c.metadata.doc_id for c in chunks)) > 1}

    if not duplicate_groups:
        print("  [OK] No cross-document duplicate chunk texts detected.")
    else:
        print(f"  [!] Found {len(duplicate_groups)} distinct text passages duplicated across multiple doc_ids:")
        for idx, (h, chunks) in enumerate(duplicate_groups.items(), start=1):
            doc_occurrences = {}
            for c in chunks:
                key = (c.metadata.doc_id, c.metadata.subject, c.metadata.filename, c.metadata.page_number)
                doc_occurrences[key] = doc_occurrences.get(key, 0) + 1

            sample_text = chunks[0].text.strip().replace("\n", " ")[:100]
            print(f"\n  Duplicate Cluster #{idx} (Hash {h[:8]}... | {len(chunks)} total instances):")
            print(f"    Snippet: \"{sample_text}...\"")
            for (d_id, subj, fname, p_num), count in sorted(doc_occurrences.items()):
                print(f"      * doc_id={d_id} | subject='{subj}' | page={p_num} | file='{fname}' ({count}x)")

    # -------------------------------------------------------------------------
    # (d) Embedding health check
    # -------------------------------------------------------------------------
    print(f"\n[SECTION D] EMBEDDING HEALTH CHECK (Dense Vector Search)")
    try:
        health_query = "Diagnostic test: university academic syllabus question answering"
        test_results = vs.collection.query(
            query_texts=[health_query],
            n_results=min(1, total_chunks)
        )
        if test_results and test_results.get("ids") and test_results["ids"][0]:
            dist = test_results["distances"][0][0] if test_results.get("distances") else 0.0
            sim = max(0.0, min(1.0, 1.0 - dist))
            print(f"  [OK] ChromaDB embedding function is healthy and operational.")
            print(f"       Test query: '{health_query}'")
            print(f"       Top distance: {dist:.4f} | Cosine similarity: {sim:.4f}")
        else:
            print("  [WARNING] ChromaDB query executed but returned empty results.")
    except Exception as e:
        import traceback
        print(f"  [FAIL] Embedding search threw an exception: {e}")
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # (e) Vector vs Hybrid Results for Target Query
    # -------------------------------------------------------------------------
    print(f"\n[SECTION E] QUERY RETRIEVAL COMPARISON")
    print(f"  Query:     \"{query}\"")
    print(f"  Subject:   {subject if subject else 'All subjects (None)'}")
    print(f"  Unit:      {unit if unit else 'All units (None)'}")
    print(f"  Top K:     {top_k}")
    print(f"  Threshold: {threshold:.4f}")

    # Mode 1: Pure Vector Search
    print(f"\n  --- Mode: DENSE VECTOR SEARCH ---")
    try:
        v_chunks = hybrid.search(query=query, subject=subject, unit=unit, mode="vector", top_k=top_k)
        print(f"  Retrieved {len(v_chunks)} chunk(s):")
        for idx, c in enumerate(v_chunks, start=1):
            snippet = c.text.strip().replace("\n", " ")[:110]
            print(f"    [{idx}] chunk_id: {c.chunk_id}")
            print(f"        doc_id: {c.metadata.doc_id} | subject: '{c.metadata.subject}' | page: {c.metadata.page_number} | unit: {c.metadata.unit}")
            print(f"        vector_score: {c.vector_score} | bm25_score: {c.bm25_score} | rrf_score: {c.rrf_score}")
            print(f"        text: \"{snippet}...\"")
        v_guardrail = evaluate(chunks=v_chunks, query=query, threshold=threshold, mode="vector")
        print(f"  Guardrail Decision: {'PASSED [OK]' if v_guardrail.passed else 'REJECTED [FALLBACK]'}")
        print(f"  Reason: {v_guardrail.reason}")
    except VectorSearchError as e:
        print(f"  [FAIL] Vector search raised VectorSearchError: {e}")
    except Exception as e:
        print(f"  [FAIL] Vector search error: {e}")

    # Mode 2: Hybrid Search
    print(f"\n  --- Mode: HYBRID SEARCH (Vector + BM25 + RRF) ---")
    try:
        h_chunks = hybrid.search(query=query, subject=subject, unit=unit, mode="hybrid", top_k=top_k)
        print(f"  Retrieved {len(h_chunks)} chunk(s):")
        for idx, c in enumerate(h_chunks, start=1):
            snippet = c.text.strip().replace("\n", " ")[:110]
            print(f"    [{idx}] chunk_id: {c.chunk_id}")
            print(f"        doc_id: {c.metadata.doc_id} | subject: '{c.metadata.subject}' | page: {c.metadata.page_number} | unit: {c.metadata.unit}")
            print(f"        vector_score: {c.vector_score} | bm25_score: {c.bm25_score} | rrf_score: {c.rrf_score}")
            print(f"        text: \"{snippet}...\"")
        h_guardrail = evaluate(chunks=h_chunks, query=query, threshold=threshold, mode="hybrid")
        print(f"  Guardrail Decision: {'PASSED [OK]' if h_guardrail.passed else 'REJECTED [FALLBACK]'}")
        print(f"  Reason: {h_guardrail.reason}")
    except Exception as e:
        print(f"  [FAIL] Hybrid search error: {e}")

    print("\n" + "=" * 80)
    print("                          END OF DIAGNOSTIC REPORT")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose RAG retrieval, ChromaDB health, and guardrail decisions.")
    parser.add_argument("query", nargs="?", default="Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation",
                        help="Question to test retrieval for (default: Nielsen 10 Heuristics)")
    parser.add_argument("--subject", default=None, help="Subject metadata filter (e.g., 'Human Computer Interaction')")
    parser.add_argument("--unit", default=None, help="Unit metadata filter (e.g., 'Unit 3' or '3')")
    parser.add_argument("--threshold", type=float, default=0.32, help="Guardrail confidence threshold (default: 0.32)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")

    args = parser.parse_args()
    diagnose(
        query=args.query,
        subject=args.subject,
        unit=args.unit,
        threshold=args.threshold,
        top_k=args.top_k
    )

