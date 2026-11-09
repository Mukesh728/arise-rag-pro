"""
MRSPTU StudyRAG - Main FastAPI Application
==========================================
Provides REST API endpoints for:
1. Document ingestion (PDF upload, chunking, Chroma vector storage, BM25 indexing)
2. Grounded question answering (Vector vs. Hybrid RRF search, citation extraction, LLM answer generation)
3. Evaluation benchmark (Vector hit rate vs. Hybrid hit rate comparison)
4. Metadata filtering and document management
"""

import os
import time
import uuid
import shutil
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.models import (
    Chunk,
    ChunkMetadata,
    Citation,
    QueryRequest,
    QueryResponse,
    IngestResponse,
    FilterOptions,
    EvaluationItem,
    EvaluationQueryInput,
    EvaluationMetricRow,
    EvaluationReport,
)
from backend.app.rag.pdf_extractor import PDFExtractor
from backend.app.rag.chunker import TokenChunker
from backend.app.rag.bm25_search import BM25Index
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.hybrid_search import HybridSearchService
from backend.app.rag.llm import GeminiAnswerGenerator, NOT_FOUND_MESSAGE


# ---------------------------------------------------------
# Service Singletons
# ---------------------------------------------------------
vector_store = ChromaVectorStore()
bm25_index = BM25Index()
hybrid_service = HybridSearchService(vector_store=vector_store, bm25_index=bm25_index)
llm_generator = GeminiAnswerGenerator()
chunker = TokenChunker(
    target_tokens=settings.CHUNK_SIZE_TOKENS,
    overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
)


# ---------------------------------------------------------
# Application Lifespan (Startup / Shutdown)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
    Synchronizes in-memory BM25 index with all chunks persisted in Chroma.
    This guarantees BM25 keyword search is immediately available without re-uploading PDFs.
    """
    try:
        existing_chunks = vector_store.get_all_chunks()
        bm25_index.build_index(existing_chunks)
        print(f"[Startup] Synchronized BM25 index with {len(existing_chunks)} chunks from ChromaDB.")
    except Exception as e:
        print(f"[Startup Warning] Could not sync BM25 index: {e}")
    yield


# ---------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Grounded Study Assistant for MRSPTU Syllabus using Hybrid RAG (Vector + BM25 + RRF + Gemini)",
    lifespan=lifespan,
)

# Enable CORS for local frontend development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Frontend Static Mounting & Health Endpoints
# ---------------------------------------------------------
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the interactive MRSPTU StudyRAG Web Interface."""
    if os.path.exists(frontend_dir):
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    return JSONResponse({
        "status": "online",
        "app_name": settings.APP_NAME,
        "message": "MRSPTU StudyRAG Backend is running. Frontend directory not found."
    })

@app.get("/spiderman_intro.mp4", include_in_schema=False)
async def serve_intro_video():
    """Serves the cinematic intro animation video."""
    if os.path.exists(frontend_dir):
        video_file = os.path.join(frontend_dir, "spiderman_intro.mp4")
        if os.path.exists(video_file):
            return FileResponse(video_file, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Intro video not found")

@app.get(f"{settings.API_PREFIX}/health", tags=["Health"])
async def health_check():
    """Return health status and total indexed chunk count."""
    total_chunks = vector_store.count()
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "total_chunks": total_chunks,
        "vector_store": settings.VECTOR_STORE_TYPE,
    }


# ---------------------------------------------------------
# Document Ingestion Endpoint
# ---------------------------------------------------------
@app.post(
    f"{settings.API_PREFIX}/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Upload and index a syllabus/notes PDF document",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    subject: str = Form(..., description="Course/Subject name, e.g. Operating Systems"),
    unit: str = Form("1", description="Unit or Chapter number, e.g. Unit 1"),
):
    """
    1. Validates uploaded PDF file.
    2. Saves PDF copy to local uploads directory.
    3. Extracts text per page preserving exact 1-indexed page numbers.
    4. Chunks text into ~500 token windows with 50 token overlap.
    5. Embeds and saves chunks into ChromaDB with metadata.
    6. Adds chunks to the in-memory BM25 index for keyword retrieval.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    clean_subject = subject.strip()
    # Accept "1", "Unit 1", "UNIT-I", "II" etc. from the form field
    raw_unit = unit.strip()
    # Extract digits or Roman numeral via the same regex the chunker uses
    import re as _re
    _UNIT_FORM_RE = _re.compile(r"UNIT[\s\-:]*([IVX]+|\d+)|^([IVX]+|\d+)$", _re.IGNORECASE)
    _ROMAN = [("M",1000),("CM",900),("D",500),("CD",400),("C",100),("XC",90),("L",50),("XL",40),("X",10),("IX",9),("V",5),("IV",4),("I",1)]
    def _r2i(s):
        s, r, i = s.upper(), 0, 0
        for n, v in _ROMAN:
            while s[i:i+len(n)] == n: r += v; i += len(n)
        return r if r > 0 else 1
    m = _UNIT_FORM_RE.search(raw_unit)
    if m:
        g = m.group(1) or m.group(2)
        clean_unit: int = int(g) if g.isdigit() else _r2i(g)
    else:
        try:
            clean_unit = int(raw_unit)
        except ValueError:
            clean_unit = 1   # safe default
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    # Read uploaded file bytes
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    # Persist file copy to disk
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    saved_filename = f"{doc_id}_{os.path.basename(file.filename)}"
    saved_path = os.path.join(settings.UPLOAD_DIR, saved_filename)
    try:
        with open(saved_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        print(f"[Ingest Warning] Failed to save copy of upload to disk: {e}")

    # Extract pages with 1-indexed page numbers
    try:
        pages_data = PDFExtractor.extract_pages_from_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting text from PDF: {str(e)}",
        )

    if not pages_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in PDF. Scanned images without OCR are not supported.",
        )

    # Chunk pages with token sliding window with automatic unit detection
    chunks = chunker.process_document(
        pages_data=pages_data,
        doc_id=doc_id,
        filename=file.filename,
        subject=clean_subject,
        unit=clean_unit,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to generate chunks from the uploaded document.",
        )

    # Add to Chroma Vector Store and BM25 Index
    try:
        vector_store.add_chunks(chunks)
        bm25_index.add_chunks(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing document chunks: {str(e)}",
        )

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        subject=clean_subject,
        unit=str(clean_unit),
        total_pages=len(pages_data),
        total_chunks=len(chunks),
        message=f"Successfully ingested '{file.filename}' into {clean_subject} (Unit {clean_unit}).",
    )


# ---------------------------------------------------------
# Query & RAG Question-Answering Endpoint
# ---------------------------------------------------------
@app.post(
    f"{settings.API_PREFIX}/query",
    response_model=QueryResponse,
    tags=["RAG Query"],
    summary="Ask a question against syllabus materials with citation tracking",
)
async def query_rag(request: QueryRequest):
    """
    Processes student query using Vector Search or Hybrid Search (Vector + BM25 + RRF).
    Extracts citations and generates a grounded Gemini answer with strict page references.
    """
    print(f"[API DEBUG] Request body: {request.dict() if hasattr(request, 'dict') else request.model_dump()}")
    start_time = time.time()
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    search_mode = request.mode.strip().lower() if request.mode else "hybrid"
    threshold = (
        request.threshold
        if request.threshold is not None
        else settings.SIMILARITY_THRESHOLD
    )
    top_k = request.top_k if request.top_k > 0 else settings.DEFAULT_TOP_K

    # Step 1: Retrieve chunks using selected search mode
    t_retrieval_start = time.time()
    try:
        retrieved_chunks = hybrid_service.search(
            query=query_text,
            subject=request.subject,
            unit=request.unit,
            mode=search_mode,
            top_k=top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}",
        )
    retrieval_time_ms = round((time.time() - t_retrieval_start) * 1000, 2)

    # Step 2: Build citations and check similarity threshold
    citations: List[Citation] = []
    max_vector_score = 0.0

    for idx, c in enumerate(retrieved_chunks, start=1):
        score = c.vector_score if c.vector_score is not None else (c.rrf_score or 0.0)
        if c.vector_score and c.vector_score > max_vector_score:
            max_vector_score = c.vector_score

        snippet = c.text[:220].strip() + ("..." if len(c.text) > 220 else "")
        # First non-empty line as key highlight
        first_line = c.text.strip().split("\n")[0]
        highlight = first_line[:120] if first_line else snippet[:120]

        citations.append(
            Citation(
                citation_id=idx,
                filename=c.metadata.filename,
                subject=c.metadata.subject,
                unit=str(c.metadata.unit) if c.metadata.unit is not None else "",
                page_number=c.metadata.page_number,
                source_snippet=snippet,
                highlighted_text=highlight,
                similarity_score=score,
            )
        )

    # Step 3: Generate grounded answer with Gemini
    t_llm_start = time.time()
    is_fallback = False
    valid_vector_scores = [
        getattr(c, 'score', getattr(c, 'similarity', getattr(c, 'vector_score', getattr(c, 'similarity_score', None))))
        for c in retrieved_chunks
    ]
    valid_vector_scores = [s for s in valid_vector_scores if s is not None]
    max_vector_score = max(valid_vector_scores) if valid_vector_scores else 0.0

    if not retrieved_chunks or max_vector_score < threshold:
        answer = NOT_FOUND_MESSAGE
        is_fallback = True
        gemini_time_ms = 0.0
    else:
        try:
            answer = llm_generator.generate_answer(
                query=query_text,
                chunks=retrieved_chunks,
                threshold=threshold,
            )
            if NOT_FOUND_MESSAGE.lower() in answer.lower():
                is_fallback = True
        except Exception as e:
            print(f"[Main Query Error] LLM generation failed: {e}")
            answer = f"Error: {e}"
            is_fallback = True
        gemini_time_ms = round((time.time() - t_llm_start) * 1000, 2)

    fallback_message = None
    if is_fallback:
        fallback_message = (
            f"The similarity score of the uploaded material did not meet the confidence threshold ({threshold}). "
            "The system prevented an ungrounded hallucination and returned the exact required fallback message:"
        )

    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    print(f"[Query Timing] Retrieval: {retrieval_time_ms}ms | LLM: {gemini_time_ms}ms | Total: {execution_time_ms}ms")

    # Serialize raw chunks for transparency
    raw_chunks_payload = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "vector_score": c.vector_score,
            "bm25_score": c.bm25_score,
            "rrf_score": c.rrf_score,
            "rank": c.rank,
            "metadata": c.metadata.model_dump(),
        }
        for c in retrieved_chunks
    ]

    return QueryResponse(
        query=query_text,
        answer=answer,
        mode=search_mode,
        citations=citations,
        retrieved_chunks=raw_chunks_payload,
        is_fallback=is_fallback,
        threshold=threshold,
        fallback_message=fallback_message,
        execution_time_ms=execution_time_ms,
        retrieval_time_ms=retrieval_time_ms,
        llm_time_ms=gemini_time_ms,
    )


# ---------------------------------------------------------
# Metadata & Filter Options Endpoint
# ---------------------------------------------------------
@app.get(
    f"{settings.API_PREFIX}/filters",
    response_model=FilterOptions,
    tags=["Metadata"],
    summary="Get distinct subjects and units for dropdown filters",
)
async def get_filter_options():
    """Returns available subjects and associated units to populate UI dropdowns."""
    try:
        data = vector_store.get_distinct_subjects_and_units()
        return FilterOptions(
            subjects=data.get("subjects", []),
            units_by_subject=data.get("units_by_subject", {}),
            total_documents=data.get("total_documents", 0),
            total_chunks=data.get("total_chunks", 0),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading filter metadata: {str(e)}",
        )


# ---------------------------------------------------------
# Evaluation Benchmark Endpoint (Vector vs Hybrid Search)
# ---------------------------------------------------------
@app.post(
    f"{settings.API_PREFIX}/evaluate",
    response_model=EvaluationReport,
    tags=["Evaluation"],
    summary="Benchmark retrieval accuracy: Vector Search vs Hybrid Search",
)
async def evaluate_rag_retrieval(eval_input: EvaluationQueryInput):
    """
    Runs each test question through both Vector Search and Hybrid Search.
    Verifies if the expected page appears in the top-K retrieved chunks (Hit Rate@K).
    Returns comparative performance metrics for viva demonstration and reports.
    """
    queries = eval_input.queries
    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Queries list for evaluation cannot be empty.",
        )

    top_k = eval_input.top_k if eval_input.top_k > 0 else settings.DEFAULT_TOP_K
    vector_hits = 0
    hybrid_hits = 0
    detailed_results: List[EvaluationMetricRow] = []

    for idx, item in enumerate(queries, start=1):
        subj = item.subject or eval_input.subject
        unit = item.unit or eval_input.unit

        # 1. Evaluate pure vector search
        v_chunks = hybrid_service.search(
            query=item.question,
            subject=subj,
            unit=unit,
            mode="vector",
            top_k=top_k,
        )
        v_pages = [c.metadata.page_number for c in v_chunks]
        v_snippets = [c.text[:120] for c in v_chunks]
        v_hit = item.expected_page in v_pages
        if v_hit:
            vector_hits += 1

        # 2. Evaluate hybrid RRF search
        h_chunks = hybrid_service.search(
            query=item.question,
            subject=subj,
            unit=unit,
            mode="hybrid",
            top_k=top_k,
        )
        h_pages = [c.metadata.page_number for c in h_chunks]
        h_snippets = [c.text[:120] for c in h_chunks]
        h_hit = item.expected_page in h_pages
        if h_hit:
            hybrid_hits += 1

        detailed_results.append(
            EvaluationMetricRow(
                id=item.id if item.id is not None else idx,
                question=item.question,
                subject=subj or "All",
                unit=unit or "All",
                expected_page=item.expected_page,
                vector_top_pages=v_pages,
                vector_hit=v_hit,
                hybrid_top_pages=h_pages,
                hybrid_hit=h_hit,
                vector_snippets=v_snippets,
                hybrid_snippets=h_snippets,
            )
        )

    total_q = len(queries)
    v_hit_rate = round(vector_hits / total_q, 4) if total_q > 0 else 0.0
    h_hit_rate = round(hybrid_hits / total_q, 4) if total_q > 0 else 0.0
    delta = round(h_hit_rate - v_hit_rate, 4)

    return EvaluationReport(
        total_queries=total_q,
        vector_hits=vector_hits,
        hybrid_hits=hybrid_hits,
        vector_hit_rate=v_hit_rate,
        hybrid_hit_rate=h_hit_rate,
        delta_improvement=delta,
        detailed_results=detailed_results,
    )


# ---------------------------------------------------------
# Document Deletion Endpoint
# ---------------------------------------------------------
@app.delete(
    f"{settings.API_PREFIX}/documents/{{doc_id}}",
    tags=["Management"],
    summary="Delete a document and its chunks from both Chroma and BM25",
)
async def delete_document(doc_id: str):
    """Removes all chunks belonging to doc_id from Chroma and re-indexes BM25."""
    try:
        vector_store.delete_document(doc_id)
        bm25_index.remove_document(doc_id)
        return {"message": f"Document '{doc_id}' successfully removed from indexes."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document {doc_id}: {str(e)}",
        )


# ---------------------------------------------------------
# Sample Syllabus Generation Endpoint (Viva & Testing)
# ---------------------------------------------------------
@app.post(
    f"{settings.API_PREFIX}/generate-sample-syllabus",
    tags=["Ingestion"],
    summary="Generate and index sample MRSPTU Operating Systems syllabus for testing and viva",
)
async def generate_sample_syllabus():
    """
    Seeds the ChromaDB vector store and BM25 index with a standard 5-page
    MRSPTU Operating Systems syllabus covering Unit 1 & Unit 2:
    - Page 1 (Unit 1): Process Management, Process States (New, Ready, Running, Waiting, Terminated), PCB, Context Switch
    - Page 2 (Unit 1): CPU Scheduling Algorithms (FCFS, SJF, Round Robin with quantum, Priority)
    - Page 3 (Unit 1): Process Synchronization, Critical Section Problem, Peterson's Solution, Semaphores, Mutex
    - Page 4 (Unit 2): Deadlocks, Coffman Conditions, Banker's Algorithm, Safety Algorithm
    - Page 5 (Unit 2): Memory Management, Paging, Page Table, TLB hit ratio and EMAT calculation, Thrashing
    """
    doc_id = "doc_mrsptu_os_sample"
    filename = "MRSPTU_Operating_Systems_Syllabus.pdf"
    subject = "Operating Systems"

    sample_pages = [
        {
            "page_number": 1,
            "unit": 1,
            "text": (
                "MRSPTU B.Tech CSE - Operating Systems (Unit 1)\n"
                "CHAPTER 1: PROCESS MANAGEMENT AND PROCESS STATES\n\n"
                "A process is defined as a program in execution. A process is more than the program code (text section); "
                "it also includes the current activity represented by the program counter and CPU registers, "
                "the stack containing temporary data (function parameters, return addresses, local variables), "
                "and a data section containing global variables.\n\n"
                "Process States: As a process executes, it changes state according to the five-state process model:\n"
                "1. New: The process is being created and loaded into memory.\n"
                "2. Ready: The process has all necessary resources allocated and is waiting to be assigned to a CPU core by the short-term scheduler.\n"
                "3. Running: Instructions are being executed by the CPU.\n"
                "4. Waiting (Blocked): The process is waiting for some event to occur (such as an I/O completion or reception of an IPC signal).\n"
                "5. Terminated: The process has finished execution and its resources are reclaimed by the operating system.\n\n"
                "Process Control Block (PCB): Each process is represented in the operating system by a PCB (Task Control Block). "
                "The PCB contains crucial metadata: Process ID (PID), Process State, Program Counter (PC), CPU registers (accumulators, index registers, stack pointers), "
                "CPU Scheduling Information (process priority, queue pointers), Memory-Management Information (page tables or segment tables), "
                "Accounting Information (CPU time used, time limits), and I/O Status Information (list of allocated I/O devices and open files).\n\n"
                "Context Switch: Switching the CPU to another process requires performing a state save of current process into its PCB "
                "and a state restore of the different process from its PCB. Context switch time is pure overhead because the system does no useful work while switching."
            ),
        },
        {
            "page_number": 2,
            "unit": "Unit 1",
            "text": (
                "MRSPTU B.Tech CSE - Operating Systems (Unit 1)\n"
                "CHAPTER 2: CPU SCHEDULING ALGORITHMS\n\n"
                "CPU scheduling is the basis of multi-programmed operating systems. By switching the CPU among processes, "
                "the OS makes the computer more productive. The short-term scheduler (CPU scheduler) selects from among the processes "
                "in ready memory and allocates the CPU to one of them.\n\n"
                "Key Scheduling Criteria: CPU utilization, Throughput, Turnaround time (time from submission to completion), "
                "Waiting time (sum of periods spent waiting in the ready queue), and Response time.\n\n"
                "1. First-Come, First-Served (FCFS) Scheduling: The process that requests the CPU first is allocated the CPU first using a FIFO queue. "
                "FCFS is strictly non-preemptive. A major disadvantage is the Convoy Effect: short processes behind a long CPU-burst process suffer high average waiting times.\n\n"
                "2. Shortest-Job-First (SJF) Scheduling: Associates each process with the length of its next CPU burst. "
                "The process with the shortest burst is assigned the CPU. SJF is proven to be optimal because it yields the minimum average waiting time. "
                "Its preemptive version is called Shortest-Remaining-Time-First (SRTF). The major difficulty is predicting the length of the next CPU burst (often estimated using exponential averaging).\n\n"
                "3. Round-Robin (RR) Scheduling: Designed specifically for time-sharing systems. A small unit of time called a Time Quantum or Time Slice (typically 10 to 100 ms) is defined. "
                "The ready queue is treated as a circular queue. The CPU scheduler goes around the queue, allocating the CPU to each process for an interval of up to 1 time quantum. "
                "If the process has a CPU burst greater than 1 time quantum, it is preempted and put at the tail of the ready queue. "
                "Performance heavily depends on quantum size: if too large, RR behaves like FCFS; if too small, context-switch overhead dominates."
            ),
        },
        {
            "page_number": 3,
            "unit": "Unit 1",
            "text": (
                "MRSPTU B.Tech CSE - Operating Systems (Unit 1)\n"
                "CHAPTER 3: PROCESS SYNCHRONIZATION & CRITICAL SECTION PROBLEM\n\n"
                "Cooperating processes can either directly share a logical address space or be permitted to share data only through files or messages. "
                "Concurrent access to shared data may result in data inconsistency (Race Condition). "
                "A Race Condition occurs when several processes access and manipulate the same data concurrently and the outcome depends on the particular order of access.\n\n"
                "The Critical Section Problem: Consider a system consisting of n processes {P0, P1, ..., Pn-1}. "
                "Each process has a segment of code, called a critical section, in which the process may be changing common variables, updating a table, or writing a file. "
                "When one process is executing in its critical section, no other process is allowed to execute in its critical section.\n\n"
                "Three Strict Requirements for Any Valid Solution:\n"
                "1. Mutual Exclusion: If process Pi is executing in its critical section, then no other processes can be executing in their critical sections.\n"
                "2. Progress: If no process is executing in its critical section and some processes wish to enter, only those processes not executing in their remainder sections can participate in deciding which will enter next, and this selection cannot be postponed indefinitely.\n"
                "3. Bounded Waiting: There must be a bound or limit on the number of times that other processes are allowed to enter their critical sections after a process has made a request to enter its critical section and before that request is granted.\n\n"
                "Peterson's Solution: A classic software-based solution for two processes. It uses two shared variables: 'int turn' and 'boolean flag[2]'. "
                "The variable turn indicates whose turn it is to enter the critical section. The flag array indicates if a process is ready to enter.\n\n"
                "Semaphores and Mutex: A Semaphore S is an integer variable accessed only through two standard atomic operations: wait() [historically P] and signal() [historically V]. "
                "Counting semaphores control access to a given resource consisting of a finite number of instances. Binary semaphores behave similarly to Mutex locks, providing mutual exclusion."
            ),
        },
        {
            "page_number": 4,
            "unit": 2,
            "text": (
                "MRSPTU B.Tech CSE - Operating Systems (Unit 2)\n"
                "CHAPTER 4: DEADLOCKS AND BANKER'S ALGORITHM\n\n"
                "In a multiprogramming environment, several processes may compete for a finite number of resources. "
                "A process requests resources; if the resources are not available at that time, the process enters a waiting state. "
                "A Deadlock occurs when every process in a set is waiting for an event that can be caused only by another process in the set.\n\n"
                "Four Coffman Conditions Necessary for Deadlock:\n"
                "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode (only one process can use the resource at any given time).\n"
                "2. Hold and Wait: A process must be currently holding at least one resource and simultaneously waiting to acquire additional resources held by other processes.\n"
                "3. No Preemption: Resources cannot be forcibly preempted; a resource can be released only voluntarily by the process holding it, after that process has completed its task.\n"
                "4. Circular Wait: A closed loop {P0, P1, ..., Pn} must exist such that P0 is waiting for a resource held by P1, P1 is waiting for P2, and Pn is waiting for P0.\n\n"
                "Deadlock Avoidance & Banker's Algorithm: Banker's algorithm is a deadlock avoidance algorithm developed by Edsger Dijkstra. "
                "When a new process enters the system, it must declare the maximum number of instances of each resource type that it may need. "
                "When a process requests a set of resources, the system must determine whether the allocation will leave the system in a Safe State.\n\n"
                "Data Structures in Banker's Algorithm:\n"
                "- Available[m]: Vector of length m indicating available instances of each resource.\n"
                "- Max[n][m]: Maximum demand of each process.\n"
                "- Allocation[n][m]: Number of resources of each type currently allocated to each process.\n"
                "- Need[n][m]: Remaining resource need of each process, calculated as: Need[i][j] = Max[i][j] - Allocation[i][j].\n"
                "Safety Algorithm determines if there exists a safe sequence <P1, P2, ..., Pn> such that for each Pi, the resources that Pi can still request can be satisfied by currently available resources plus resources held by all preceding processes Pj (j < i)."
            ),
        },
        {
            "page_number": 5,
            "unit": 2,
            "text": (
                "MRSPTU B.Tech CSE - Operating Systems (Unit 2)\n"
                "CHAPTER 5: MEMORY MANAGEMENT, PAGING AND VIRTUAL MEMORY\n\n"
                "Memory consists of a large array of bytes, each with its own address. The CPU fetches instructions from memory according to the value of the program counter.\n\n"
                "Logical vs. Physical Address Space: An address generated by the CPU is commonly referred to as a Logical Address (Virtual Address). "
                "An address loaded into the memory address register (MAR) of the physical memory hardware is referred to as a Physical Address. "
                "The run-time mapping from virtual to physical addresses is done by a hardware device called the Memory Management Unit (MMU).\n\n"
                "Paging: Paging is a memory-management scheme that permits the physical address space of a process to be non-contiguous. "
                "Paging completely avoids external fragmentation and the need for compaction. "
                "Physical memory is divided into fixed-sized blocks called Frames (typically 4 KB). Logical memory is divided into blocks of the same size called Pages. "
                "Every address generated by the CPU is divided into two parts: a Page Number (p) and a Page Offset (d). "
                "The Page Table is used as an index into physical frame numbers.\n\n"
                "Translation Lookaside Buffer (TLB): Accessing memory via page table requires two memory accesses (one for page table entry, one for actual byte). "
                "To speed up paging, a small, fast-lookup hardware associative cache called the TLB is used.\n\n"
                "Effective Memory Access Time (EMAT) Formula:\n"
                "Let h be the TLB Hit Ratio (percentage of times a page number is found in the TLB).\n"
                "Let t_tlb be the TLB access/search time (e.g. 20 ns) and t_mem be the main memory access time (e.g. 100 ns).\n"
                "EMAT = h * (t_tlb + t_mem) + (1 - h) * (t_tlb + 2 * t_mem)\n"
                "If hit ratio is 80%, EMAT = 0.80 * (20 + 100) + 0.20 * (20 + 200) = 96 + 44 = 140 ns.\n\n"
                "Thrashing: If a process does not have enough frames allocated, the page-fault rate becomes very high. "
                "A process is thrashing if it is spending more time paging (swapping pages in and out of backing store) than executing instructions. "
                "Thrashing can be prevented using the Working-Set Model based on the principle of locality."
            ),
        },
    ]

    # Delete existing sample if already present to allow clean re-seed
    try:
        vector_store.delete_document(doc_id)
        bm25_index.remove_document(doc_id)
    except Exception:
        pass

    created_chunks: List[Chunk] = []
    for p in sample_pages:
        page_num = p["page_number"]
        unit_str = p["unit"]
        text_content = p["text"]
        chunk_id = f"{doc_id}_p{page_num}_c0"

        meta = ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=doc_id,
            filename=filename,
            subject=subject,
            unit=int(p["unit"]),
            page_number=page_num,
            char_start=0,
            char_end=len(text_content),
            token_count=len(text_content.split()),
        )
        created_chunks.append(Chunk(chunk_id=chunk_id, text=text_content, metadata=meta))

    vector_store.add_chunks(created_chunks)
    bm25_index.add_chunks(created_chunks)

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": filename,
        "subject": subject,
        "pages_created": len(sample_pages),
        "chunks_indexed": len(created_chunks),
        "message": f"Successfully indexed sample syllabus '{filename}' ({len(created_chunks)} pages/chunks) into ChromaDB and BM25 index.",
    }
