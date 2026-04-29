import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.models.schemas import IngestURLRequest, IngestResponse
from app.services.pinecone_service import get_pinecone_service
from app.core.security import verify_api_key
from app.core.logging import get_logger
from ingestion.pipeline import ingest_urls, ingest_pdf

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)


@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    request: IngestURLRequest,
    _: str = Depends(verify_api_key),
):
    """Scrape URLs, chunk content, upsert into Pinecone."""
    try:
        chunks = ingest_urls(request.urls, source_type=request.source_type)
        if not chunks:
            raise HTTPException(status_code=422, detail="No content extracted from URLs")
        ids = get_pinecone_service().upsert_documents(chunks)
        return IngestResponse(
            message="URLs ingested successfully",
            chunks_ingested=len(chunks),
            vector_ids=ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("URL ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf_file(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    """Upload a PDF, chunk it, upsert into Pinecone."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        chunks = ingest_pdf(tmp_path, source_type="pdf")
        for chunk in chunks:
            chunk.metadata["original_filename"] = file.filename

        ids = get_pinecone_service().upsert_documents(chunks)
        return IngestResponse(
            message=f"PDF '{file.filename}' ingested successfully",
            chunks_ingested=len(chunks),
            vector_ids=ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)