from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

@lru_cache
def get_embeddings() ->OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance
    Returns:
        Configured OpenAIEmbedding instance
    """
    
    settings = get_settings()
    logger.info(f"Initializing embedding model: {settings.embedding_model}")
    
    embeddings = OpenAIEmbeddings(
        model = settings.embedding_model,
        openai_api_key = settings.openai_api_key
    )
    
    logger.info("Embedding model initialize successfully")
    return embeddings

class EmbeddingService:
    "Service for general embeddings"
    
    def __init__(self):
        
        settings = get_settings()
        self.embedding = get_embeddings()
        self.model_name = settings.embedding_model
        
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for user/single query"""
        
        logger.info(f"Embedding user query: {text[:50]}...")
        return self.embedding.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        "Generate embeddings for documents"
        logger.info(f"Generate embeddings for {len(texts)} documents")
        
        return self.embedding.embed_documents(texts)