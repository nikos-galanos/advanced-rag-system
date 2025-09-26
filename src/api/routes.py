"""
FastAPI routes for the RAG system.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import time
from datetime import datetime

from .models import (
    UploadResponse, 
    QueryRequest, 
    QueryResponse, 
    ErrorResponse
)
from src.core.ingestion import DocumentProcessor

router = APIRouter()

# Initialize document processor
doc_processor = DocumentProcessor()


@router.post("/ingest", response_model=UploadResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest one or more PDF documents into the knowledge base.
    """
    start_time = time.time()
    
    try:
        # Validate files
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF"
                )
        
        # Process documents using the real DocumentProcessor
        total_chunks = await doc_processor.process_documents(files)
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            message="Documents processed successfully",
            files_processed=len(files),
            chunks_created=total_chunks,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base and generate an answer.
    """
    start_time = time.time()
    
    return QueryResponse(
        query=request.query,
        answer=f"WORKING NOW! You asked: '{request.query}'. Advanced processing will be implemented soon.",
        citations=[],
        metadata={
            "status": "working_correctly",
            "available_chunks": doc_processor.get_chunk_count(),
            "confirmation": "artifact_updated_successfully"
        },
        processing_time=time.time() - start_time,
        timestamp=datetime.now()
    )


@router.get("/stats")
async def get_system_stats():
    """
    Get system statistics.
    """
    try:
        stats = {
            "total_documents": doc_processor.get_document_count(),
            "total_chunks": doc_processor.get_chunk_count(),
            "index_size": 0,  # TODO: implement in retrieval engine
            "last_updated": doc_processor.get_last_update()
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))