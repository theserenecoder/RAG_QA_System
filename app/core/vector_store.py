"""Vectore Store Module for Qdrant Operations"""

from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

EMBEDDING_DOMENSION = 1536


@lru_cache
def get_qdrant_client() ->QdrantClient:
    """Get cached qdrant client instance

    Returns:
        QdrantClient: _description_
    """
    logger.info(f"Connecting to qdrant at : {settings.qdrant_url}")
    
    client  =QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )
    
    logger.info("Qdrant client connected succesfully")
    
    return client


class VectorStoreService:
    """Services to manage vector store operations"""
    
    def __init__(self, collection_name: str | None = None):
        """Initialize vectore store services

        Args:
            collection_name (str | None, optional): Name of the qdrant collection
        """
        
        self.collection_name = collection_name or settings.collection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        
        ## ensure collection exists
        self._ensure_collection()
        
        ## Initialize Langchain Qdrant vectore store
        self.vectore_store = QdrantVectorStore(
            client = self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        logger.info(f"VectoreStoreServices initialized for collection : {self.collection_name}")
         
    def _ensure_collection(self)->None:
        """Ensure collection exists"""
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}' exists with"
                f"{collection_info.points_count} points"
            )
        except UnexpectedResponse:
            
            logger.info(f"Creating Collection : {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size = EMBEDDING_DOMENSION,
                    distance=Distance.COSINE
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")           
            
    def add_document(self, documents: list[Document]) ->list[str]:
        """Add documents to vector store

        Args:
            documents (list[Document]): List of Document objects to add

        Returns:
            List of document ids
        """
        
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to collection")
        
        ## generating unique ids for each documents
        ids = [str(uuid4()) for _ in documents]
        
        ## add to vectore store
        self.vectore_store.add_documents(documents=documents, ids=ids)
        
        logger.info(f"Successfully added {len(documents)} documents")
        
        return ids   
    
    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Search for similar documents

        Args:
            query (str): Search query
            k (int | None, optional): Number of results to return

        Returns:
            List of document objects
        """
        
        k = k or settings.retrieval_k
        logger.debug(f"Searching for : {query[:50]}...{k:k}")
        
        results = self.vectore_store.similarity_search(query,k)
        
        logger.debug(f"Found {len(results)} results")
        
        return results  
    
    def search_with_score(self, query: str, k: int | None = None) -> list[tuple[Document,float]]:
        """Search for similar documents with similarity score

        Args:
            query (str): Search query
            k : Number of results to return

        Returns:
            List of (document, score) objects
        """
        
        k = k or settings.retrieval_k
        logger.debug(f"Searching for : {query[:50]}...(k={k})")
        
        results = self.vectore_store.similarity_search_with_score(query,k)
        
        logger.debug(f"Found {len(results)} results with score")
        
        return results 
    
    def get_retriever(self, k: int | None = None) ->Any:
        """Retriever for vectore store

        Args:
            k : number of documents to retrieve

        Returns:
            LangChain retriever object
        """
        k=k or settings.retrieval_k
        
        result = self.vectore_store.as_retriever(
            search_type = 'similarity',
            search_kwargs = {'k': k}
        )
        
        return result  
    
    def delete_collection(self) ->None:
        """Delete the entire collection"""
        
        logger.warning(f"Deleting collection : {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")
        
    def get_collection_info(self) -> dict:
        """Get information about the collection

        Returns:
            Dictionary with collection statistics
        """
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,        ## how many documents are available within the collection, earlier was vector_count
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }
        except UnexpectedResponse:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "indexed_vectors_count": 0,
                "status": "not_found",
            }
            
    def health_check(self) -> bool:
        """Check if vector store is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collection()
            return True
        except Exception as e:
            logger.error(f"Vectore store health check failed: {e}")
            return False