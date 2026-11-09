from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any, Union

class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    subject: str
    unit: Optional[int] = None           # integer (e.g. 1, 2, 3) or None if no unit
    page_number: int
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    vector_score: Optional[float] = None
    score: Optional[float] = None
    similarity: Optional[float] = None
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rank: Optional[int] = None

class Citation(BaseModel):
    citation_id: int
    filename: str
    subject: str
    unit: Optional[str] = ""
    page_number: int
    source_snippet: str
    highlighted_text: str
    similarity_score: Optional[float] = None

class QueryRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    subject: Optional[str] = None
    subject_filter: Optional[str] = None
    unit: Optional[Union[str, int]] = None
    unit_filter: Optional[Union[str, int]] = None
    mode: Optional[str] = "hybrid" # "vector" or "hybrid"
    strategy: Optional[str] = "hybrid"
    top_k: int = 3
    threshold: Optional[float] = 0.32

    @model_validator(mode="before")
    @classmethod
    def reconcile_query_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Reconcile query vs question
            q = data.get("query") or data.get("question") or ""
            data["query"] = q
            data["question"] = q

            # Reconcile mode vs strategy
            s = data.get("mode") or data.get("strategy") or "hybrid"
            m = "vector" if str(s).strip().lower() == "vector" else "hybrid"
            data["mode"] = m
            data["strategy"] = s

            # Reconcile subject / subject_filter
            sub = data.get("subject") if data.get("subject") is not None else data.get("subject_filter")
            data["subject"] = sub
            data["subject_filter"] = sub

            # Reconcile unit / unit_filter
            u = data.get("unit") if data.get("unit") is not None else data.get("unit_filter")
            data["unit"] = u
            data["unit_filter"] = u

            # Top_k conversion
            if "top_k" in data and data["top_k"] is not None:
                try:
                    data["top_k"] = int(data["top_k"])
                except Exception:
                    data["top_k"] = 3
            elif "top_chunks" in data and data["top_chunks"] is not None:
                try:
                    data["top_k"] = int(data["top_chunks"])
                except Exception:
                    data["top_k"] = 3

            # Threshold conversion
            if "threshold" in data and data["threshold"] is not None:
                try:
                    data["threshold"] = float(data["threshold"])
                except Exception:
                    data["threshold"] = 0.32
        return data

class QueryResponse(BaseModel):
    query: str
    answer: str
    mode: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]
    is_fallback: bool = False
    threshold: Optional[float] = None
    fallback_message: Optional[str] = None
    execution_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    llm_time_ms: float = 0.0

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    subject: str
    unit: str
    total_pages: int
    total_chunks: int
    message: str

class FilterOptions(BaseModel):
    subjects: List[str]
    units_by_subject: Dict[str, List[str]]
    total_documents: int
    total_chunks: int

class EvaluationItem(BaseModel):
    id: Optional[int] = None
    question: str
    expected_page: int
    subject: Optional[str] = None
    unit: Optional[str] = None

class EvaluationQueryInput(BaseModel):
    queries: List[EvaluationItem]
    subject: Optional[str] = None
    unit: Optional[str] = None
    top_k: int = 3

class EvaluationMetricRow(BaseModel):
    id: int
    question: str
    subject: str
    unit: str
    expected_page: int
    vector_top_pages: List[int]
    vector_hit: bool
    hybrid_top_pages: List[int]
    hybrid_hit: bool
    vector_snippets: List[str]
    hybrid_snippets: List[str]

class EvaluationReport(BaseModel):
    total_queries: int
    vector_hits: int
    hybrid_hits: int
    vector_hit_rate: float
    hybrid_hit_rate: float
    delta_improvement: float
    detailed_results: List[EvaluationMetricRow]
