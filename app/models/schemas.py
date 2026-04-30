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