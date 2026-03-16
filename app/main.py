"""FastAPI application entry point"""


from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import documents, health, query
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

settings = get_settings()

## defining app lifespan manager, startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    ## startup
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Log level: {settings.log_level}")
    
    yield
    
    ## shutdown
    logger.info("Shutting down application")

## create fast api application
app = FastAPI(
    title=settings.app_name,
    description="""
    ## RAG Q&A System API

    A Retrieval-Augmented Generation (RAG) question-answering system built with:
    - **FastAPI** for the API layer
    - **LangChain** for RAG orchestration
    - **Qdrant Cloud** for vector storage
    - **OpenAI** for embeddings and LLM

    ### Features
    - Upload PDF, TXT, and CSV documents
    - Ask questions and get AI-powered answers
    - View source documents for transparency
    - Streaming responses for real-time feedback
    """,
    
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

## add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## include routes
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": __version__,
        "docs": "/docs",
    }
    
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gobal exception handler"""
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )
    
    
if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        "app.main:app", 
        host=settings.api_host, 
        port=settings.api_port, 
        reload=True
    )