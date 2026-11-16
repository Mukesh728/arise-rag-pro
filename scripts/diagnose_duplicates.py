import os
import sys
import hashlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.rag.vector_store import ChromaVectorStore

def diagnose():
    vs = ChromaVectorStore()
    all_chunks = vs.get_all_chunks()
    print(f"Total chunks in ChromaDB: {len(all_chunks)}")

    # Group by doc_id
    docs = defaultdict(list)
    for c in all_chunks:
        docs[c.metadata.doc_id].append(c)

    print("\n--- CURRENT STORED DOCUMENTS ---")
    kept_hci_doc_id = None
    for doc_id, chunk_list in sorted(docs.items()):
        meta = chunk_list[0].metadata
        print(f"doc_id: {doc_id} | filename: '{meta.filename}' | subject: '{meta.subject}' | chunks: {len(chunk_list)}")
        if meta.subject == "Human Computer Interaction" and "Human Computer Interaction" in meta.filename:
            kept_hci_doc_id = doc_id

    print(f"\nKept HCI doc_id candidate: {kept_hci_doc_id}")
    if kept_hci_doc_id:
        # Sort chunks by page_number and char_start
        kept_chunks = sorted(docs[kept_hci_doc_id], key=lambda c: (c.metadata.page_number, c.metadata.char_start))
        kept_hashes = [hashlib.md5(c.text.strip().encode("utf-8")).hexdigest() for c in kept_chunks]
        print(f"Kept HCI chunk count: {len(kept_chunks)}, hash count: {len(kept_hashes)}")

        print("\n--- VERIFYING UUID CANDIDATES AGAINST KEPT HCI COPY ---")
        for doc_id, chunk_list in sorted(docs.items()):
            if doc_id == kept_hci_doc_id:
                continue
            meta = chunk_list[0].metadata
            # Sort candidate chunks
            cand_chunks = sorted(chunk_list, key=lambda c: (c.metadata.page_number, c.metadata.char_start))
            cand_hashes = [hashlib.md5(c.text.strip().encode("utf-8")).hexdigest() for c in cand_chunks]
            
            is_identical = (cand_hashes == kept_hashes)
            print(f"Candidate doc_id: {doc_id} | filename: '{meta.filename}' | subject: '{meta.subject}' | chunks: {len(cand_chunks)} | Identical to Kept: {is_identical}")

    # Also list files in ./uploads
    print("\n--- FILES IN ./uploads ---")
    if os.path.exists("./uploads"):
        for f in os.listdir("./uploads"):
            print(f)
    else:
        print("./uploads does not exist")

if __name__ == "__main__":
    diagnose()
