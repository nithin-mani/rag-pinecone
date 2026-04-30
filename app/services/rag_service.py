from typing import Optional, List
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pinecone_service import get_pinecone_service
from app.services.cache_service import get_cache_service

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions
using only the provided context. If the context does not contain enough
information to answer the question, say so explicitly.
Do not make up facts. Always be concise.

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGService:
    def __init__(self):
        self.settings = get_settings()

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        namespace_filter: Optional[dict] = None,
    ) -> dict:
        k = top_k or self.settings.top_k
        logger.info("RAG query", question=question[:80])

        # ── Step 1: Check cache ──────────────────────
        cached = get_cache_service().get(question, k)
        if cached:
            cached["from_cache"] = True
            return cached

        # ── Step 2: Retrieve from Pinecone ───────────
        docs = get_pinecone_service().similarity_search(
            query=question,
            top_k=k,
            score_threshold=0.50,
            filter=namespace_filter,
        )

        if not docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "context_docs": [],
                "from_cache": False,
            }

        # ── Step 3: Format context ───────────────────
        context = _format_docs(docs)

        # ── Step 4: Call LLM ─────────────────────────
        llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            temperature=self.settings.openai_temperature,
            api_key=self.settings.openai_api_key,
        )
        chain = QA_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "input": question})

        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        result = {
            "answer": answer,
            "sources": sources,
            "context_docs": [
                {
                    "content": doc.page_content[:300],
                    "metadata": doc.metadata,
                }
                for doc in docs
            ],
            "from_cache": False,
        }

        # ── Step 5: Store in cache ───────────────────
        get_cache_service().set(question, k, result)

        return result


@lru_cache()
def get_rag_service() -> RAGService:
    return RAGService()