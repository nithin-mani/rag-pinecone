"""
RAGAS Evaluation Service
Measures RAG pipeline quality across 3 metrics:
- Faithfulness: does the answer stick to the context?
- Answer Relevancy: does the answer address the question?
- Context Recall: did we retrieve the right chunks?
"""
from typing import List, Optional
from functools import lru_cache

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pinecone_service import get_pinecone_service

import asyncio
import concurrent.futures

logger = get_logger(__name__)


class EvalService:
    def __init__(self):
        self.settings = get_settings()

    def _get_llm(self):
        return ChatOpenAI(
            model=self.settings.openai_chat_model,
            temperature=0.0,
            api_key=self.settings.openai_api_key,
        )

    def _get_embeddings(self):
        return OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key,
        )

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        logger.info("Evaluating single QA pair", question=question[:60])

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision]
        llm = self._get_llm()
        embeddings = self._get_embeddings()

        # Run in thread to avoid nested event loop conflict with uvloop
        def _run():
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run)
            result = future.result(timeout=120)

        scores = result.to_pandas().to_dict(orient="records")[0]
        cleaned = {
            "faithfulness": round(float(scores.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(scores.get("answer_relevancy", 0)), 4),
            "context_precision": round(float(scores.get("context_precision", 0)), 4),
        }
        logger.info("Evaluation complete", scores=cleaned)
        return cleaned

    def evaluate_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> dict:
        logger.info("Evaluating dataset", size=len(questions))

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision]
        llm = self._get_llm()
        embeddings = self._get_embeddings()

        def _run():
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run)
            result = future.result(timeout=120)

        df = result.to_pandas()
        rows = df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_dict(orient="records")
        averages = {
            "faithfulness": round(float(df["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
            "context_precision": round(float(df["context_precision"].mean()), 4),
        }
        logger.info("Dataset evaluation complete", averages=averages)
        return {
            "averages": averages,
            "rows": rows,
            "total_evaluated": len(questions),
        }

    def run_rag_and_evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> dict:
        """
        Run the full RAG pipeline on a list of questions
        then evaluate the results automatically.
        """
        from app.services.rag_service import get_rag_service

        rag = get_rag_service()
        answers = []
        contexts = []

        for q in questions:
            result = rag.query(question=q)
            answers.append(result["answer"])
            contexts.append([doc["content"] for doc in result["context_docs"]])

        return self.evaluate_dataset(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )


@lru_cache()
def get_eval_service() -> EvalService:
    return EvalService()