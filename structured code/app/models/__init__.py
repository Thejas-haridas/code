"""
Data models for the RAG-Enhanced SQL Query Generator and Analyzer.

This module contains Pydantic models used for request/response validation
and data structure definitions throughout the application.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for SQL query generation."""
    question: str
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is the total premium collected in 2024?"
            }
        }

class QueryResponse(BaseModel):
    """Response model for SQL query generation and analysis."""
    success: bool
    question: str
    retrieved_tables: List[str]
    generated_sql: str
    sql_execution_result: dict
    llm_analysis: str
    retrieval_time: float
    sql_generation_time: float
    llm_analysis_time: float
    sql_execution_time: float
    total_processing_time: float
    file_saved: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "question": "What is the total premium collected in 2024?",
                "retrieved_tables": ["dwh.fact_premium", "dwh.dim_policy"],
                "generated_sql": "SELECT SUM(total_paid) FROM dwh.fact_premium...",
                "sql_execution_result": {"success": True, "data": [{"total": 1000000}]},
                "llm_analysis": "The total premium collected in 2024 is $1,000,000.",
                "retrieval_time": 0.15,
                "sql_generation_time": 1.23,
                "llm_analysis_time": 0.87,
                "sql_execution_time": 0.45,
                "total_processing_time": 2.70,
                "file_saved": "/home/text_sql/query_20240101-120000.json"
            }
        }

class SQLOnlyRequest(BaseModel):
    """Request model for SQL-only generation."""
    question: str
    
class SQLOnlyResponse(BaseModel):
    """Response model for SQL-only generation."""
    question: str
    retrieved_tables: List[str]
    generated_sql: str
    generation_time: float

class RetrievalTestRequest(BaseModel):
    """Request model for testing retrieval system."""
    question: str

class RetrievalTestResponse(BaseModel):
    """Response model for retrieval testing."""
    question: str
    retrieved_tables: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    sql_model_loaded: bool
    analysis_model_loaded: bool
    rag_retriever_loaded: bool
    device: str

class MemoryStatusResponse(BaseModel):
    """Response model for memory status."""
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_reserved: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_memory_mb: float

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    sql_model: str
    analysis_model: str
    embedding_model: str
    sql_model_loaded: bool
    analysis_model_loaded: bool
    rag_system_loaded: bool

class SchemaInfoResponse(BaseModel):
    """Response model for schema information."""
    total_tables: int
    tables: List[str]
    embeddings_created: bool

class TableChunk(BaseModel):
    """Model for table schema chunks used in RAG."""
    type: str
    table: str
    text: str
    description: str
    columns: List[str]
    keywords: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "type": "table",
                "table": "dwh.dim_claims",
                "text": "Table: dwh.dim_claims | Columns: claim_reference_id, date_claim_first_notified...",
                "description": "Contains claim metadata and lifecycle events including claim status, dates, amounts, and denial information.",
                "columns": ["claim_reference_id", "date_claim_first_notified", "date_of_loss_from"],
                "keywords": ["claims", "claim", "loss", "damage", "settlement"]
            }
        }

class SQLExecutionResult(BaseModel):
    """Model for SQL execution results."""
    success: Optional[bool] = None
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    execution_time: Optional[float] = None

# Export all models
__all__ = [
    "QueryRequest",
    "QueryResponse", 
    "SQLOnlyRequest",
    "SQLOnlyResponse",
    "RetrievalTestRequest",
    "RetrievalTestResponse",
    "HealthResponse",
    "MemoryStatusResponse",
    "ModelInfoResponse",
    "SchemaInfoResponse",
    "TableChunk",
    "SQLExecutionResult"
]