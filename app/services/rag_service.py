from typing import Optional, List
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pinecone_service import get_pinecone_service

logger = get_logger(__name__)

# ─── Prompt ───────────────────────────────────────────────────
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
    """Combine retrieved docs into a single context string."""
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
        logger.info("RAG query", question=question[:80])

        # Step 1: Retrieve relevant docs from Pinecone
        docs = get_pinecone_service().similarity_search(
            query=question,
            top_k=top_k,
            filter=namespace_filter,
        )

        # Step 2: Handle no results
        if not docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "context_docs": [],
            }

        # Step 3: Format docs into context string
        context = _format_docs(docs)

        # Step 4: Call LLM with context + question
        llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            temperature=self.settings.openai_temperature,
            api_key=self.settings.openai_api_key,
        )

        chain = QA_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "input": question})

        # Step 5: Collect sources
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        logger.info("Answer generated", sources=sources)

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": [
                {
                    "content": doc.page_content[:300],
                    "metadata": doc.metadata,
                }
                for doc in docs
            ],
        }


@lru_cache()
def get_rag_service() -> RAGService:
    return RAGService()