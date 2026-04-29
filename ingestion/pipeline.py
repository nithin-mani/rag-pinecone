import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _make_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _add_chunk_id(docs: List[Document]) -> List[Document]:
    """Add a unique ID to each chunk based on content hash."""
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        doc.metadata["chunk_id"] = content_hash
    return docs


def ingest_urls(urls: List[str], source_type: str = "web") -> List[Document]:
    """Scrape web pages and chunk them."""
    logger.info("Loading URLs", count=len(urls))
    loader = WebBaseLoader(urls)
    raw_docs = loader.load()

    for doc in raw_docs:
        doc.metadata["source_type"] = source_type

    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)
    chunks = _add_chunk_id(chunks)

    logger.info("URL ingestion complete", raw=len(raw_docs), chunks=len(chunks))
    return chunks


def ingest_pdf(file_path: str, source_type: str = "pdf") -> List[Document]:
    """Load a PDF and chunk it."""
    logger.info("Loading PDF", path=file_path)
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    for doc in raw_docs:
        doc.metadata["source_type"] = source_type
        doc.metadata["source"] = file_path

    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)
    chunks = _add_chunk_id(chunks)

    logger.info("PDF ingestion complete", raw=len(raw_docs), chunks=len(chunks))
    return chunks


def ingest_text(text: str, metadata: Optional[dict] = None) -> List[Document]:
    """Chunk a raw text string."""
    meta = metadata or {}
    meta.setdefault("source_type", "text")
    doc = Document(page_content=text, metadata=meta)

    splitter = _make_splitter()
    chunks = splitter.split_documents([doc])
    chunks = _add_chunk_id(chunks)

    logger.info("Text ingestion complete", chunks=len(chunks))
    return chunks