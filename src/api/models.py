"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class UploadResponse(BaseModel):
    """File upload response."""
    message: str
    files_processed: int
    chunks_created: int
    processing_time: float


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    include_metadata: Optional[bool] = Field(True, description="Include chunk metadata")


class ChunkMetadata(BaseModel):
    """Chunk metadata model."""
    chunk_id: str
    document_name: str
    page_number: Optional[int]
    chunk_index: int
    similarity_score: float
    keyword_score: Optional[float]
    final_score: float


class Citation(BaseModel):
    """Citation model."""
    text: str
    source: str
    page: Optional[int]
    confidence: float


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    citations: List[Citation]
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str]
    timestamp: datetime