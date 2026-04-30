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

@router.get("/debug/redis")
async def debug_redis(_: str = Depends(verify_api_key)):
    """Check Redis connection status."""
    import redis as redis_lib
    from app.core.config import get_settings
    settings = get_settings()
    try:
        client = redis_lib.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=2,
        )
        client.ping()
        return {"status": "connected", "redis_url": settings.redis_url[:30]}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@router.get("/debug/cache-test")
async def debug_cache_test(_: str = Depends(verify_api_key)):
    """Test Redis cache set and get directly."""
    from app.services.cache_service import get_cache_service
    cache = get_cache_service()
    
    # Test set
    cache.set("test question", 5, {"answer": "test answer", "sources": [], "context_docs": [], "from_cache": False})
    
    # Test get
    result = cache.get("test question", 5)
    
    if result:
        return {"status": "cache working", "result": result}
    else:
        return {"status": "cache NOT working"}