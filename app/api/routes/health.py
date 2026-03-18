"""Health check endpoints"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from app import __version__
from app.api.schemas import HealthResponse, ReadinessResponse
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status of the services",
)
async def health_check() ->HealthResponse:
    """Basic health check endpoint"""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=__version__
    )
    
    
@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Check if service is ready to handle the request",
)
async def readiness_check() -> ReadinessResponse:
    """Rediness check including database connectivity."""
    logger.debug("Readiness check requested")
    
    
    try:
        # check qdrant connection
        vector_store = VectorStoreService()
        is_healthy = vector_store.health_check()
        
        if not is_healthy:
            raise HTTPException(
                status_code=503,
                detail="Vectore store is not healthy"
            )
            
        collection_info = vector_store.get_collection_info()
        
        return ReadinessResponse(
            status = "ready",
            qdrant_connected=True,
            collection_info=collection_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}",
        )
        
