from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service
from app.core.security import verify_api_key
from app.core.logging import get_logger

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger(__name__)


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: str = Depends(verify_api_key),
):
    """
    Conversational RAG endpoint.
    Pass the same session_id across turns to maintain history.
    """
    try:
        result = get_chat_service().chat(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            namespace_filter=request.namespace_filter,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error("Chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def clear_session(
    session_id: str,
    _: str = Depends(verify_api_key),
):
    """Clear conversation history for a session."""
    get_chat_service().clear_history(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/{session_id}/history")
async def get_history(
    session_id: str,
    _: str = Depends(verify_api_key),
):
    """Get conversation history for a session."""
    history = get_chat_service().get_history(session_id)
    return {
        "session_id": session_id,
        "turns": len(history) // 2,
        "messages": [
            {
                "role": "human" if isinstance(m, __import__('langchain_core.messages', fromlist=['HumanMessage']).HumanMessage) else "ai",
                "content": m.content,
            }
            for m in history
        ],
    }