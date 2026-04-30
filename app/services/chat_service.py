from typing import List, Optional, Dict
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pinecone_service import get_pinecone_service
from app.services.cache_service import get_cache_service

logger = get_logger(__name__)

# In-memory session store
# Key: session_id, Value: list of messages
_sessions: Dict[str, List] = {}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions
using only the provided context and conversation history.
If the context does not contain enough information, say so explicitly.
Do not make up facts. Be concise.

Context:
{context}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


def _format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class ChatService:
    def __init__(self):
        self.settings = get_settings()

    def get_history(self, session_id: str) -> List:
        """Get conversation history for a session."""
        return _sessions.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info("Session cleared", session_id=session_id)

    def chat(
        self,
        question: str,
        session_id: str,
        top_k: Optional[int] = None,
        namespace_filter: Optional[dict] = None,
    ) -> dict:
        k = top_k or self.settings.top_k
        logger.info("Chat query", session_id=session_id, question=question[:80])

        # Step 1: Get conversation history
        history = self.get_history(session_id)

        # Step 2: Retrieve relevant docs from Pinecone
        docs = get_pinecone_service().similarity_search(
            query=question,
            top_k=k,
            score_threshold=0.50,
            filter=namespace_filter,
        )

        context = _format_docs(docs) if docs else "No relevant context found."

        # Step 3: Call LLM with history + context
        llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            temperature=self.settings.openai_temperature,
            api_key=self.settings.openai_api_key,
        )

        chain = CHAT_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "history": history,
            "input": question,
        })

        # Step 4: Update session history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        _sessions[session_id] = history

        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        logger.info("Chat answer generated", session_id=session_id)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "turn": len(history) // 2,
        }


@lru_cache()
def get_chat_service() -> ChatService:
    return ChatService()