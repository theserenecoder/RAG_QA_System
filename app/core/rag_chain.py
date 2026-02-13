"""RAG chain module using LangChain"""

import asyncio

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger
from app.core.vector_store import VectorStoreService

logger = get_logger(__name__)
settings  = get_settings()


## RAG_TEMPLATE
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""

def format_docs(docs: list[Document]) -> str:
    """Format document into single document string

    Args:
        docs (list[Document]): List of Document objects

    Returns:
        Formatted context dtring
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """RAG chain for question answering"""
    
    def __init__(self, vector_store_services: VectorStoreService | None = None) :
        """Initialize RAG chain

        Args:
            vector_store_services : optional VectorDtoreService instance
        """
        ## Initialize vector store
        self.vector_store = vector_store_services or VectorStoreService()
        ## initialize retriever
        self.retriever = self.vector_store.get_retriever()
        
        ## Initialize evaluator
        self._evaluator = None
        
        ## Initialize LLM
        self.llm = ChatOpenAI(
            model = settings.llm_model,
            temperature= settings.llm_temperature,
            openai_api_key = settings.openai_api_key
        )
        
        ## create prompt template
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        ## build LCEL
        
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info(
            f"RAGChain initialize with model: {settings.llm_model},"
            f"retrieval_k: {settings.retrieval_k}"
        )
        
    @property
    def evaluator(self):
        """Get or create RAGAS evaluator instance"""
        if self._evaluator is None:
            from app.core.ragas_evaluator import RAGASEvaluator
            
            self._evaluator = RAGASEvaluator()
        return self._evaluator
        
    def query(self,question: str) ->str:
        """Execute a RAG query

        Args:
            question : User question

        Returns:
            Generated answer
        """
        
        logger.info("Processing query: {question[:50]...}")
        
        try:
            answer = self.chain.invoke(question)
            logger.info("Query processed successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
        
    
    def query_with_sources(self,question: str) ->dict:
        """Execute a RAG query

        Args:
            question : User question

        Returns:
            Dictionary with generated answer and source documents
        """
        try:
            answer = self.chain.invoke(question)
        
            source_docs = self.retriever.invoke(question)
            
            sources = [
                {
                    "content": (
                        doc.page_content[:500]+"..."
                        if len(doc.page_content)>500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]
            
            logger.info(f"Query processed with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing query with source: {e}")
            raise
    
    async def aquery(self,question: str) ->str:
        """Execute a RAG query

        Args:
            question : User question

        Returns:
            Generated answer
        """
        
        logger.info("Processing query: {question[:50]...}")
        
        try:
            answer = await self.chain.ainvoke(question)
            logger.info("Query processed successfully")
            return answer
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
        
    async def aquery_with_sources(self,question: str) ->dict:
        """Execute a RAG query

        Args:
            question : User question

        Returns:
            Dictionary with generated answer and source documents
        """
        try:
            answer = await self.chain.ainvoke(question)
        
            source_docs = await self.retriever.ainvoke(question)
            
            sources = [
                {
                    "content": (
                        doc.page_content[:500]+"..."
                        if len(doc.page_content)>500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]
            
            logger.info(f"Query processed with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing query with source: {e}")
            raise
        
        
    async def aquery_with_evaluation(self, question: str, included_sources: bool = True) -> dict:
        """Execute async RAG query wityh RAGAD evaluation

        Args:
            question (str): user question
            included_sources (bool, optional): Weather to include sources in response

        Returns:
            Dictionary with answer, sources and evaluation scores
        """
        logger.info(f"Processing query with evalution: {question[:100]}...")
        
        try:
            ## Get answer with source
            result = await self.aquery_with_sources(question)
            answer = result["answer"]
            sources = result["sources"]
            
            ## preparing context for evaluation
            contexts = [source["content"] for source in sources]
            
            ## Run evaluation
            try:
                evaluation = await self._evaluator.aevaluate(question, answer, contexts)     
                
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness= {evaluation.get("faithfulness","N/A")}, "
                    f"answer_relevancy= {evaluation.get("answer_relevancy","N/A")}"
                )        
            
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}", exc_info=True)
                evaluation = {
                    
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "evaluation_time_ms": None,
                    "error": str(e)
                
                }
                
            return {"answer": answer, "sources": sources, "evaluation": evaluation}
            
        except Exception as e:
            logger.error(f"Error in query with evaluation: {e}")
            raise
            
        
    def stream(self, question: str):
        """Stream RAG response.

        Args:
            question: User question

        Yields:
            Response chunks
        """
        
        try:
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise