from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import (
    EvalSingleRequest, EvalSingleResponse, EvalScores,
    EvalDatasetRequest, EvalDatasetResponse,
)
from app.services.eval_service import get_eval_service
from app.core.security import verify_api_key
from app.core.logging import get_logger

router = APIRouter(prefix="/eval", tags=["Evaluation"])
logger = get_logger(__name__)


@router.post(
    "/single",
    response_model=EvalSingleResponse,
    summary="Evaluate a single QA pair",
    description="Pass a question, answer, and retrieved contexts. Returns RAGAS scores.",
)
async def eval_single(
    request: EvalSingleRequest,
    _: str = Depends(verify_api_key),
):
    try:
        scores = get_eval_service().evaluate_single(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth,
        )
        return EvalSingleResponse(
            question=request.question,
            scores=EvalScores(**scores),
        )
    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/dataset",
    response_model=EvalDatasetResponse,
    summary="Run RAG pipeline and evaluate automatically",
    description="Pass a list of questions. The system runs RAG on each and evaluates the results.",
)
async def eval_dataset(
    request: EvalDatasetRequest,
    _: str = Depends(verify_api_key),
):
    try:
        result = get_eval_service().run_rag_and_evaluate(
            questions=request.questions,
            ground_truths=request.ground_truths,
        )
        return EvalDatasetResponse(
            total_evaluated=result["total_evaluated"],
            averages=EvalScores(**result["averages"]),
            rows=result["rows"],
        )
    except Exception as e:
        logger.error("Dataset evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))