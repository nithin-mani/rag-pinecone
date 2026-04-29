from typing import List, Optional
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PineconeService:
    def __init__(self):
        self.settings = get_settings()
        self._pc = None
        self._embeddings = None
        self._vector_store = None

    # ─── Connect ──────────────────────────────────────
    def _get_pinecone_client(self) -> Pinecone:
        if self._pc is None:
            self._pc = Pinecone(api_key=self.settings.pinecone_api_key)
            logger.info("Pinecone client initialised")
        return self._pc

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
            )
        return self._embeddings

    # ─── Index ────────────────────────────────────────
    def ensure_index_exists(self) -> None:
        pc = self._get_pinecone_client()
        existing = [idx.name for idx in pc.list_indexes()]

        if self.settings.pinecone_index_name not in existing:
            logger.info("Creating Pinecone index", index=self.settings.pinecone_index_name)
            pc.create_index(
                name=self.settings.pinecone_index_name,
                dimension=self.settings.pinecone_embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment,
                ),
            )
            logger.info("Index created successfully")
        else:
            logger.info("Index already exists", index=self.settings.pinecone_index_name)

    def get_vector_store(self) -> PineconeVectorStore:
        if self._vector_store is None:
            self.ensure_index_exists()
            pc = self._get_pinecone_client()
            index = pc.Index(self.settings.pinecone_index_name)
            self._vector_store = PineconeVectorStore(
                index=index,
                embedding=self._get_embeddings(),
                namespace=self.settings.pinecone_namespace,
            )
            logger.info("PineconeVectorStore ready")
        return self._vector_store

    # ─── Write ────────────────────────────────────────
    def upsert_documents(self, documents: List[Document]) -> List[str]:
        vs = self.get_vector_store()
        ids = vs.add_documents(documents)
        logger.info("Upserted documents", count=len(ids))
        return ids

    # ─── Read ─────────────────────────────────────────
    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        k = top_k or self.settings.top_k
        threshold = score_threshold or self.settings.similarity_threshold
        vs = self.get_vector_store()

        results = vs.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

        filtered = [doc for doc, score in results if score >= threshold]
        logger.info(
            "Similarity search",
            total=len(results),
            after_threshold=len(filtered),
        )
        return filtered

    def get_retriever(self, top_k: Optional[int] = None, filter: Optional[dict] = None):
        vs = self.get_vector_store()
        search_kwargs = {"k": top_k or self.settings.top_k}
        if filter:
            search_kwargs["filter"] = filter
        return vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    # ─── Stats ────────────────────────────────────────
    def get_index_stats(self) -> dict:
        pc = self._get_pinecone_client()
        index = pc.Index(self.settings.pinecone_index_name)
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "namespaces": {
                ns: info.vector_count
                for ns, info in (stats.namespaces or {}).items()
            },
            "dimension": stats.dimension,
        }


@lru_cache()
def get_pinecone_service() -> PineconeService:
    return PineconeService()