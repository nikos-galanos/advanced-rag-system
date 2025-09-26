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
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)
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
    Query the knowledge base and generate an answer using enhanced processing.
    """
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from src.core.query import QueryProcessor
        
        # Initialize enhanced query processor
        query_processor = QueryProcessor()
        
        # Build context for query transformation
        context = {
            "document_count": doc_processor.get_document_count(),
            "chunk_count": doc_processor.get_chunk_count()
        }
        
        # Process query with advanced intent detection and transformation
        processed_query = await query_processor.process_query(request.query, context)
        
        # Check if query should be refused
        should_refuse, refusal_reason = query_processor.should_refuse_query(
            request.query, 
            processed_query
        )
        
        if should_refuse:
            return QueryResponse(
                query=request.query,
                answer=f"I cannot process this query. {refusal_reason}",
                citations=[],
                metadata={
                    "status": "refused",
                    "reason": refusal_reason,
                    "intent": processed_query.get("intent"),
                    "detection_method": processed_query.get("detection_method")
                },
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        # If greeting, respond appropriately without search
        if processed_query.get("intent") == "greeting" or not processed_query.get("trigger_search"):
            greeting_responses = {
                "greeting": "Hello! I'm your advanced document assistant with hybrid query processing. I can help you find information from uploaded documents using both semantic and keyword search. What would you like to know?",
                "conversational": "I understand. Is there anything specific you'd like to know from the documents?"
            }
            
            answer = greeting_responses.get(processed_query.get("intent"), 
                                         "I'm here to help you find information from documents using advanced retrieval. What would you like to know?")
            
            return QueryResponse(
                query=request.query,
                answer=answer,
                citations=[],
                metadata={
                    "status": "no_search_needed",
                    "intent": processed_query.get("intent"),
                    "query_type": processed_query.get("query_type"),
                    "confidence": processed_query.get("confidence"),
                    "detection_method": processed_query.get("detection_method"),
                    "available_chunks": doc_processor.get_chunk_count()
                },
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        # For search queries, show enhanced processing results
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            answer=f"✨ ENHANCED PROCESSING WORKING! Intent: {processed_query.get('intent')}, "
                   f"Type: {processed_query.get('query_type')}, "
                   f"Transformed: '{processed_query.get('transformed_query')}', "
                   f"Variations: {len(processed_query.get('search_variations', []))}. "
                   f"Full retrieval implementation coming next in Step 4!",
            citations=[],
            metadata={
                "status": "enhanced_processing_active",
                "intent": processed_query.get("intent"),
                "query_type": processed_query.get("query_type"),
                "confidence": processed_query.get("confidence"),
                "detection_method": processed_query.get("detection_method"),
                "trigger_search": processed_query.get("trigger_search"),
                
                # Show transformation results
                "original_query": processed_query.get("original_query"),
                "cleaned_query": processed_query.get("cleaned_query"),
                "transformed_query": processed_query.get("transformed_query"),
                "synonym_expanded": processed_query.get("synonym_expanded"),
                "llm_enhanced": processed_query.get("llm_enhanced"),
                "search_variations": processed_query.get("search_variations"),
                "query_keywords": processed_query.get("query_keywords"),
                "transformation_strategy": processed_query.get("transformation_strategy"),
                "search_strategy": processed_query.get("search_strategy"),
                
                "available_chunks": doc_processor.get_chunk_count(),
                "enhanced_features_active": True
            },
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


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