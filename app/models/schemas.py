from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# ─── Query ────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    namespace_filter: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is retrieval augmented generation?",
                "top_k": 5,
            }
        }
    }


class ContextDoc(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    context_docs: List[ContextDoc]
    question: str
    from_cache: bool = False


# ─── Ingest ───────────────────────────────────────────
class IngestURLRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1)
    source_type: str = Field("web")

    model_config = {
        "json_schema_extra": {
            "example": {
                "urls": ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"],
                "source_type": "wikipedia",
            }
        }
    }


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int
    vector_ids: List[str]


# ─── Health ───────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str = "1.0.0"


class IndexStatsResponse(BaseModel):
    total_vector_count: int
    namespaces: Dict[str, int]
    dimension: int


# ─── Chat ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(..., description="Unique session ID for conversation history")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    namespace_filter: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is RAG?",
                "session_id": "user-123",
                "top_k": 5,
            }
        }
    }


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    turn: int


# ─── Evaluation ───────────────────────────────────────
class EvalSingleRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is RAG?",
                "answer": "RAG combines retrieval with generation.",
                "contexts": ["RAG is a method that retrieves documents..."],
                "ground_truth": "RAG stands for Retrieval Augmented Generation.",
            }
        }
    }


class EvalScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float


class EvalSingleResponse(BaseModel):
    question: str
    scores: EvalScores


class EvalDatasetRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "questions": [
                    "What is RAG?",
                    "What are the benefits of RAG?",
                ],
                "ground_truths": [
                    "RAG stands for Retrieval Augmented Generation.",
                    "RAG improves accuracy by grounding answers in retrieved documents.",
                ],
            }
        }
    }


class EvalDatasetResponse(BaseModel):
    total_evaluated: int
    averages: EvalScores
    rows: List[dict]