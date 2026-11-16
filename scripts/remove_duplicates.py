import os
import sys
import hashlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.bm25_search import BM25Index
from backend.app.config import settings

def remove_duplicates():
    vs = ChromaVectorStore()
    bm = BM25Index()
    bm.build_index(vs.get_all_chunks())

    all_chunks = vs.get_all_chunks()
    docs = defaultdict(list)
    for c in all_chunks:
        docs[c.metadata.doc_id].append(c)

    kept_hci_doc_id = "doc_051a25df"
    assert kept_hci_doc_id in docs, f"Kept HCI doc {kept_hci_doc_id} not found!"
    kept_chunks = sorted(docs[kept_hci_doc_id], key=lambda c: (c.metadata.page_number, c.metadata.char_start))
    kept_hashes = [hashlib.md5(c.text.strip().encode("utf-8")).hexdigest() for c in kept_chunks]

    duplicates_to_delete = []
    for doc_id, chunk_list in docs.items():
        if doc_id == kept_hci_doc_id:
            continue
        meta = chunk_list[0].metadata
        if meta.filename.endswith(".pdf") and len(meta.filename.replace(".pdf", "")) == 36: # UUID length
            cand_chunks = sorted(chunk_list, key=lambda c: (c.metadata.page_number, c.metadata.char_start))
            cand_hashes = [hashlib.md5(c.text.strip().encode("utf-8")).hexdigest() for c in cand_chunks]
            if cand_hashes == kept_hashes:
                duplicates_to_delete.append(doc_id)
            else:
                raise ValueError(f"Candidate {doc_id} has UUID filename but hashes do NOT match kept copy!")

    print(f"Found {len(duplicates_to_delete)} verified duplicate documents to delete: {duplicates_to_delete}")
    assert len(duplicates_to_delete) == 6, f"Expected 6 duplicates, found {len(duplicates_to_delete)}"

    for doc_id in duplicates_to_delete:
        print(f"\nDeleting {doc_id}...")
        # 1. Delete from ChromaDB
        vs.delete_document(doc_id)
        # 2. Delete from BM25
        bm.remove_document(doc_id)
        # 3. Remove upload file
        matching_uploads = [
            f for f in os.listdir(settings.UPLOAD_DIR)
            if f.startswith(f"{doc_id}_")
        ]
        for f in matching_uploads:
            f_path = os.path.join(settings.UPLOAD_DIR, f)
            print(f"Removing upload file: {f_path}")
            os.remove(f_path)

    # Verification
    print("\n--- POST-DELETION VERIFICATION ---")
    remaining_chunks = vs.get_all_chunks()
    bm_chunks = bm.chunks
    print(f"Remaining chunks in ChromaDB: {len(remaining_chunks)}")
    print(f"Remaining chunks in BM25: {len(bm_chunks)}")
    assert len(remaining_chunks) == 36, f"Expected 36 chunks in ChromaDB, found {len(remaining_chunks)}"
    assert len(bm_chunks) == 36, f"Expected 36 chunks in BM25, found {len(bm_chunks)}"

    remaining_docs = defaultdict(list)
    for c in remaining_chunks:
        remaining_docs[c.metadata.doc_id].append(c)

    print(f"Total remaining documents: {len(remaining_docs)}")
    assert len(remaining_docs) == 3, f"Expected 3 documents, found {len(remaining_docs)}"

    for doc_id, chunk_list in sorted(remaining_docs.items()):
        meta = chunk_list[0].metadata
        print(f"Kept doc: {doc_id} | '{meta.filename}' | '{meta.subject}' | chunks: {len(chunk_list)}")
        assert "08e7acfe" not in meta.subject, f"UUID subject still found in {doc_id}"

    distinct_info = vs.get_distinct_subjects_and_units()
    print("Distinct subjects in ChromaDB:", distinct_info["subjects"])
    assert len(distinct_info["subjects"]) == 3, f"Expected 3 subjects, found {len(distinct_info['subjects'])}"
    for subj in distinct_info["subjects"]:
        assert "08e7acfe" not in subj, f"UUID subject still present: {subj}"

    print("\nSUCCESS: All duplicate documents cleanly removed and verified!")

if __name__ == "__main__":
    remove_duplicates()
