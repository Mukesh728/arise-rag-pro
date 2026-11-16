import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.llm import GeminiAnswerGenerator, MODEL_NAME, NOT_FOUND_MESSAGE

vs = ChromaVectorStore()
all_chunks = vs.get_all_chunks()

c1 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 1][0]
c13 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 13][0]
c10 = [c for c in all_chunks if c.metadata.doc_id == 'doc_051a25df' and c.metadata.page_number == 10][0]

chunks = [c1, c13, c10]
query = "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation"

context_parts = []
for idx, chunk in enumerate(chunks, start=1):
    page_num = chunk.metadata.page_number
    subj = chunk.metadata.subject
    unit = chunk.metadata.unit
    context_parts.append(f"[Source Chunk {idx} | Subject: {subj} | Unit: {unit} | Page: {page_num}]\n{chunk.text}")
context_text = "\n\n---\n\n".join(context_parts)

prompt = f"""You are a helpful and precise university academic tutor assisting a student with their course syllabus.

IMPORTANT RULES:
1. Answer the student's question STRICTLY and ONLY using the provided reference material below.
2. Do NOT extrapolate or assume facts not present in the reference material.
3. If the reference material does not contain relevant information to answer the question, you MUST respond exactly with:
   "{NOT_FOUND_MESSAGE}"
4. If the reference material contains partial information (for example, if only some of the items or heuristics are covered in the retrieved text), explain all available items thoroughly from the provided text citing page numbers.
5. When explaining the answer, cite the page number(s) (e.g., "[Page {chunks[0].metadata.page_number}]") so the student can verify it in their textbook/notes.
6. Keep your tone academic, clear, and easy to understand for viva preparation.

--- REFERENCE MATERIAL ---
{context_text}  
--------------------------

STUDENT QUESTION:
{query}

ANSWER:"""

llm = GeminiAnswerGenerator()
resp = llm.client.models.generate_content(model=MODEL_NAME, contents=prompt)
print("UPDATED PROMPT RESPONSE:")
print(resp.text)

