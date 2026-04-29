from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, ContextDoc
from app.services.rag_service import get_rag_service
from app.core.security import verify_api_key
from app.core.logging import get_logger

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    _: str = Depends(verify_api_key),
):
    """Submit a question — returns an LLM answer grounded in retrieved docs."""
    try:
        result = get_rag_service().query(
            question=request.question,
            top_k=request.top_k,
            namespace_filter=request.namespace_filter,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            context_docs=[ContextDoc(**d) for d in result["context_docs"]],
            question=request.question,
        )
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))