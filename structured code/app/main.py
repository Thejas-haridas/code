"""
FastAPI Application Entry Point
"""
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import os

from rag_with_faiss import RAGSystem
from models import QueryRequest, QueryResponse
from utils import setup_device, cleanup_memory
from schema import TABLE_CHUNKS, EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting RAG-Enhanced SQL Query Generator...")
    
    # Initialize device and RAG system
    app.state.device = setup_device()
    app.state.rag_system = RAGSystem(TABLE_CHUNKS, EMBEDDING_MODEL_NAME, app.state.device)
    
    # Load models
    await app.state.rag_system.initialize()
    app.state.loop = asyncio.get_event_loop()
    
    logger.info("✅ RAG system and models loaded successfully!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    cleanup_memory()
    executor.shutdown(wait=True)
    logger.info("🧹 Cleanup completed.")

# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG-Enhanced SQL Query Generator and Analysis API",
    version="3.0.0",
    description="Generate and analyze SQL queries using RAG with FAISS-powered schema retrieval",
    lifespan=lifespan
)

@app.post("/generate-and-analyze-sql", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main endpoint to generate SQL, execute it, and analyze results."""
    try:
        result = await app.state.rag_system.process_complete_query(
            request.question, 
            app.state.loop, 
            executor
        )
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/generate-sql-only")
async def generate_sql_only(request: QueryRequest):
    """Generate SQL query without execution or analysis using RAG."""
    try:
        result = await app.state.rag_system.generate_sql_only(
            request.question, 
            app.state.loop, 
            executor
        )
        return result
    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")

@app.post("/test-retrieval")
async def test_retrieval(request: QueryRequest):
    """Test the RAG retrieval system independently."""
    try:
        result = await app.state.rag_system.test_retrieval(request.question)
        return result
    except Exception as e:
        logger.error(f"Retrieval test error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval test failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": app.state.rag_system.are_models_loaded() if hasattr(app.state, 'rag_system') else False,
        "device": str(app.state.device) if hasattr(app.state, 'device') else "unknown"
    }

@app.get("/memory-status")
async def memory_status():
    """Get current memory usage statistics."""
    if hasattr(app.state, 'rag_system'):
        return app.state.rag_system.get_memory_status()
    return {"error": "RAG system not initialized"}

@app.get("/model-info")
async def model_info():
    """Get information about loaded models."""
    if hasattr(app.state, 'rag_system'):
        return app.state.rag_system.get_model_info()
    return {"error": "RAG system not initialized"}

@app.get("/schema-info")
async def schema_info():
    """Get information about the available schema tables."""
    if hasattr(app.state, 'rag_system'):
        return app.state.rag_system.get_schema_info()
    return {"error": "RAG system not initialized"}

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "RAG-Enhanced SQL Generation and Analysis API is running",
        "version": "3.0.0",
        "features": [
            "FAISS-powered schema retrieval",
            "SQL generation with Llama-3-SQLCoder",
            "Analysis with Phi-3-mini",
            "T-SQL optimization"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )