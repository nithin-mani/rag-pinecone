from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse, IndexStatsResponse
from app.services.pinecone_service import get_pinecone_service
from app.core.config import get_settings
from app.core.security import verify_api_key

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Public health check — no auth required."""
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
    )


@router.get("/stats", response_model=IndexStatsResponse)
async def index_stats(_: str = Depends(verify_api_key)):
    """Returns total vector count and per-namespace breakdown."""
    stats = get_pinecone_service().get_index_stats()
    return IndexStatsResponse(**stats)

@router.post("/debug/search")
async def debug_search(
    question: str,
    _: str = Depends(verify_api_key),
):
    """Raw Pinecone search with scores — for debugging only."""
    from app.services.pinecone_service import get_pinecone_service
    vs = get_pinecone_service().get_vector_store()
    results = vs.similarity_search_with_score(query=question, k=5)
    return [
        {
            "score": float(score),
            "content": doc.page_content[:200],
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]