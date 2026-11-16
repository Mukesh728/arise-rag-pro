import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore

vs = ChromaVectorStore()

# 1. Test Chroma where filter
print("Testing Chroma where filter with various subjects:")
for s in ["Human Computer Interaction", "human computer interaction", "HUMAN COMPUTER INTERACTION", "Human Computer Interaction ", "08e7acfe 9a17 4877 89fb 9ab9012de5de"]:
    try:
        where = vs._build_where_filter(subject=s)
        res = vs.collection.query(query_texts=["Nielsen"], n_results=3, where=where)
        print(f"Subject '{s}': where={where} -> {len(res['ids'][0])} results")
    except Exception as e:
        print(f"Subject '{s}': EXCEPTION -> {e}")

# 2. Test Chroma where filter with unit
print("\nTesting Chroma where filter with units:")
for u in [1, 3, "1", "3", "Unit 1", "Unit 3", "unit 3", "Unit-3"]:
    try:
        where = vs._build_where_filter(subject="Human Computer Interaction", unit=u)
        res = vs.collection.query(query_texts=["Nielsen"], n_results=3, where=where)
        print(f"Unit '{u}': where={where} -> {len(res['ids'][0])} results")
    except Exception as e:
        print(f"Unit '{u}': EXCEPTION -> {e}")

# 3. Test what happens when where matches NOTHING
print("\nTesting Chroma where filter when 0 documents match:")
try:
    where = vs._build_where_filter(subject="NonExistentSubject")
    res = vs.collection.query(query_texts=["Nielsen"], n_results=3, where=where)
    print(f"NonExistent: where={where} -> {len(res['ids'][0])} results")
except Exception as e:
    print(f"NonExistent: EXCEPTION -> {type(e).__name__}: {e}")

