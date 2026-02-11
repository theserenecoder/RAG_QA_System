"""Pydantic Schemas for API request/response models"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

#================ Health Schema =======================

class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory= datetime.now, description= "Response timestamp")
    version: str = Field(..., description="Application version")
    
class ReadinessResponse(BaseModel):
    """Rediness check response."""
    
    status: str = Field(..., description="Service status")
    qdrant_connected: str = Field(..., description="Qdrant connection status")
    collection_info: dict = Field(..., description="Collection information")
    
    
#================ Documents Schema =======================

class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of chunks created")
    document_ids: list[str] = Field(..., description="List of dicument IDs")
    
    
class DocumentInfo(BaseModel):
    """Document Information."""
    
    source: str = Field(..., description="Document source/filename")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    

class DocumentListResponse(BaseModel):
    """Response for listing document"""
    
    collection_name: str = Field(..., description="Collection name")
    total_documents: int = Field(..., description="Total document count")
    status: str = Field(..., description="Collection status")
    

#================ Query Schema =======================


class QueryRequest(BaseModel):
    """Request for RAG query"""
    
    question: str = Field(..., description="Question to ask", min_length=3, max_length=2000)
    include_sources: bool = Field(default=True, description="Include source document in response")
    enable_evaluation: bool = Field(default=False, description="Enable RAGAS evaluation")
    model_config = {
        "json_schema_extra":{
            "examples": [
                {
                    "question": "What is RAG",
                    "include_source": True,
                    "enable_evaluation": False
                    
                }
            ]
        }
    }
    
    
class SourceDocument(BaseModel):
    """Source document information."""

    content: str = Field(..., description="Document content excerpt")
    metadata: dict[str, Any] = Field(..., description="Document metadata")


class EvaluationScores(BaseModel):
    """RAGAS evaluation scores."""

    faithfulness: float | None = Field(
        None,
        description="Faithfulness score (0-1): measures factual consistency with sources",
        ge=0.0,
        le=1.0,
    )
    answer_relevancy: float | None = Field(
        None,
        description="Answer relevancy score (0-1): measures relevance to question",
        ge=0.0,
        le=1.0,
    )
    evaluation_time_ms: float | None = Field(
        None,
        description="Time taken for evaluation in milliseconds",
    )
    error: str | None = Field(
        None,
        description="Error message if evaluation failed",
    )
    
    
class QueryResponse(BaseModel):
    """Response for RAG query"""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated response")
    sources: list[SourceDocument] | None = Field(None, description="Source document used")
    processing_time_ms: float = Field(..., description="Query processing time in milisecond")
    evaluation: EvaluationScores | None = Field(..., description="RAGAS evaluation scores (if requested)")