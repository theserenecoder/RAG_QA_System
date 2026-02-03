"""RAGAS Evaluation modile for RAG quality assessment"""

import asyncio
import time
from typing import Any

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """Evaluator for RAG responses using RAGAS metrics"""
    
    def __init__(self):
        """Initialize RAGAS evaluator with metrics and models"""
        
        logger.info("Initializing RAGAS Evaluator")
        self.settings = get_settings()
        
        eval_llm_model = self.settings.ragas_llm_model or self.settings.llm_model
        eval_temperature = self.settings.ragas_llm_temperature or self.settings.llm_temperature
        eval_embedding_model = self.settings.ragas_embedding_model or self.settings.embedding_model
        
        ## Initializing llm for evaluation
        self.eval_llm = ChatOpenAI(
            model=eval_llm_model,
            temperature=eval_temperature,
            openai_api_key = self.settings.openai_api_key
        )
        
        ## Initializing embedding for evaluation
        self.eval_embedding = OpenAIEmbeddings(
            model=eval_embedding_model,
            openai_api_key = self.settings.openai_api_key
        )
        
        ## Initializing metrics
        self.metrics = [
            faithfulness,
            answer_relevancy,
        ]
        
        logger.info(
            f"RAGAS evaluatior initialized..."
            f"LLM: {eval_llm_model} (temp={eval_temperature})"
            f"Embeddings: {eval_embedding_model}"
            f"Metrics: {[metric.name for metric in self.metrics]}"
        )
        
    async def aevaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Execute async RAGAS evaluation.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dictionary with evaluation scores and metadata
        """
        logger.info("Starting Evaluation")
        start_time = time.time()
        
        
        
        try:
            ## Prepare dataset for RAGAS
            dataset = self._prepare_dataset(question,answer,contexts)
            
            ## Run evaluation in thread pool to avoid blocking event loop
            results = await asyncio.to_thread(
                self._evaluate_with_timeout,
                dataset
            )
            
            evaluation_time_ms = (time.time() - start_time)*1000
            
            ## extract scores
            scores = {
                "faithfulness": float(results["faithfulness"]) if "faithfulness" in results else None,
                "answer_relevancy":float(results["answer_relevancy"]) if "answer_relevancy" in results else None,
                "evaluation_time_ms" : round(evaluation_time_ms,2),
                "error": None,
            }
            
            if self.settings.ragas_log_results:
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness= {scores["faithfulness"]},"
                    f"answer_relevancy= {scores["answer_relevancy"]},"
                    f"time= {scores['evaluation_time_ms']} ms"
                )
            return scores
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return self._handle_evaluation_error(e)
    
    def _prepare_dataset(
        self,
        question: str,
        answer: str,
        contexts: str
    ) -> Dataset:
        
        """Convert RAG output to RAGAS Dataset format.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dataset object for RAGAS evaluation
        """
        
        data = {
            "question": [question],
            "answer": [answer],
            "context": [contexts], #List of lists
        }
        
        logger.debug(f"Prepared dataset with {len(contexts)} contexts for question: {question[:50]}")
        
        return Dataset.from_dict(data)
        
    def _evaluate_with_timeout(self, dataset: Dataset) ->dict[str,Any]:
        """Execute RAGAS evaluation with timeout.

        Args:
            dataset: Prepared RAGAS dataset

        Returns:
            Evaluation results dictionary

        Raises:
            TimeoutError: If evaluation exceeds timeout
        """
        # Note: asyncio.timeout would be ideal, but RAGAS evaluate() is sync
        # For now, we rely on the async wrapper and trust RAGAS to complete
        # In production, consider using signal.alarm or threading.Timer
        
        result = evaluate(
            dataset,
            metrics= self.metrics,
            llm= self.eval_llm,
            embeddings= self.eval_embedding,
        )
        
        ## conver the result to dectionary
        return result.to_pandas().to_dict("records")[0]
    
    def _handle_evaluation_error(self, error: Exception) -> dict[str, Any]:
        """Return safe fallback scores on error.

        Args:
            error: The exception that occurred

        Returns:
            Dictionary with null scores and error message
        """
        logger.error(f"Returning fallback scores due to error: {error}")

        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(error),
        }