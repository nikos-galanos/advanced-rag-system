"""
FastAPI routes for the complete RAG system with full retrieval and generation pipeline.
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
from src.core.retrieval import HybridRetrievalEngine
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize components
doc_processor = DocumentProcessor()
retrieval_engine = HybridRetrievalEngine(doc_processor)


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
        
        # Process documents using the DocumentProcessor
        total_chunks = await doc_processor.process_documents(files)
        
        # Build indices for new documents
        if total_chunks > 0:
            logger.info("Building search indices for newly ingested documents...")
            index_stats = await retrieval_engine.build_indices(force_rebuild=True)
            logger.info(f"Index build completed: {index_stats.get('status', 'unknown')}")
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            message=f"Documents processed successfully. {total_chunks} chunks created and indexed for search.",
            files_processed=len(files),
            chunks_created=total_chunks,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in document ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base using complete hybrid retrieval and generation pipeline.
    """
    start_time = time.time()
    
    try:
       
        from src.core.query import QueryProcessor
        from src.core.generation import GenerationEngine
        
        # Initialize components
        query_processor = QueryProcessor()
        generation_engine = GenerationEngine()
        
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
                "greeting": "Hello! I'm your advanced document assistant with hybrid semantic and keyword search capabilities. I can help you find information from uploaded documents. What would you like to know?",
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
        
        # Check if we have documents to search
        if doc_processor.get_chunk_count() == 0:
            return QueryResponse(
                query=request.query,
                answer="I don't have any documents in my knowledge base yet. Please upload some PDF documents first",
                citations=[],
                metadata={
                    "status": "no_documents",
                    "available_chunks": 0,
                    "suggestion": "Upload PDF documents using the /ingest endpoint"
                },
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        # Perform hybrid retrieval using the complete pipeline
        logger.info(f"Performing hybrid retrieval for query: {processed_query.get('intent')} - {processed_query.get('query_type')}")
        
        search_results = await retrieval_engine.retrieve(processed_query, top_k=request.top_k or 5)
        
        # Check if retrieval found relevant results
        if not search_results:
            return QueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the uploaded documents that matches your query. The documents may not contain information about this topic, or you might want to try rephrasing your question.",
                citations=[],
                metadata={
                    "status": "no_results_found",
                    "intent": processed_query.get("intent"),
                    "query_type": processed_query.get("query_type"),
                    "search_performed": True,
                    "available_chunks": doc_processor.get_chunk_count(),
                    "retrieval_attempted": True
                },
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        # Generate answer using retrieved chunks and complete generation pipeline
        logger.info(f"Generating answer from {len(search_results)} retrieved chunks using GenerationEngine")
        
        generation_response = await generation_engine.generate_answer(
            query=processed_query,
            search_results=search_results,
            include_metadata=request.include_metadata
        )
        
        processing_time = time.time() - start_time
        
        # Return complete response with full pipeline results
        return QueryResponse(
            query=request.query,
            answer=generation_response["answer"],
            citations=generation_response["citations"],
            metadata={
                "status": "success",
                "intent": processed_query.get("intent"),
                "query_type": processed_query.get("query_type"),
                "confidence": processed_query.get("confidence"),
                "detection_method": processed_query.get("detection_method"),
                "trigger_search": processed_query.get("trigger_search"),
                
                # Enhanced transformation results
                "transformation_results": {
                    "original_query": processed_query.get("original_query"),
                    "transformed_query": processed_query.get("transformed_query"),
                    "synonym_expanded": processed_query.get("synonym_expanded"),
                    "llm_enhanced": processed_query.get("llm_enhanced"),
                    "search_variations": processed_query.get("search_variations"),
                    "query_keywords": processed_query.get("query_keywords"),
                    "transformation_strategy": processed_query.get("transformation_strategy")
                },
                
                # Retrieval results with detailed metrics
                "retrieval_results": {
                    "chunks_retrieved": len(search_results),
                    "top_semantic_score": max([r.semantic_score for r in search_results]) if search_results else 0,
                    "top_keyword_score": max([r.keyword_score for r in search_results]) if search_results else 0,
                    "top_combined_score": max([r.combined_score for r in search_results]) if search_results else 0,
                    "search_strategy": processed_query.get("search_strategy"),
                    "chunks_with_tables": sum(1 for r in search_results if r.chunk.metadata.get("contains_structured_data", False)),
                    "avg_similarity": sum([r.combined_score for r in search_results]) / len(search_results) if search_results else 0
                },
                
                # Generation metadata
                "generation_metadata": generation_response.get("metadata", {}),
                "available_chunks": doc_processor.get_chunk_count(),
                "table_statistics": doc_processor.get_table_statistics()
            },
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in complete query processing pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@router.get("/stats")
async def get_system_stats():
    """
    Get comprehensive system statistics with dynamic capabilities detection.
    """
    try:
        retrieval_stats = retrieval_engine.get_retrieval_statistics()
        table_stats = doc_processor.get_table_statistics()
        
        # Dynamically detect active pipeline components
        pipeline_components = []
        advanced_features = []
        
        # Document processing capabilities
        if doc_processor.get_chunk_count() > 0:
            pipeline_components.append("PDF document ingestion with intelligent chunking")
            
        if table_stats["total_tables_detected"] > 0:
            pipeline_components.append("Table detection and structured data processing")
            advanced_features.append("Table awareness with metadata enhancement")
        
        # Retrieval capabilities
        if retrieval_stats["embeddings_stats"].get("cached_embeddings", 0) > 0:
            pipeline_components.append("Semantic search with Mistral embeddings")
            advanced_features.append("Vector similarity search")
            
        if retrieval_stats["keyword_stats"].get("status") == "active":
            pipeline_components.append("BM25 keyword search implementation")
            advanced_features.append(f"Keyword index with {retrieval_stats['keyword_stats'].get('unique_terms', 0)} terms")
        
        if retrieval_stats["index_built"]:
            pipeline_components.append("Hybrid retrieval with MMR re-ranking")
            advanced_features.append("Multi-modal search combination")
        
        # Query processing capabilities (check if LLM client is configured)
        try:
            from src.utils.llm_client import MistralClient
            llm_client = MistralClient()
            if llm_client.api_key:
                pipeline_components.append("LLM-powered intent detection and query transformation")
                advanced_features.append("Domain-aware synonym expansion")
                advanced_features.append("Query refusal policies (PII/medical/legal)")
        except:
            pass
        
        # Generation capabilities
        pipeline_components.append("Context-aware answer generation")
        pipeline_components.append("Citation system with confidence scoring")
        advanced_features.append("Evidence-based generation with hallucination filtering")
        advanced_features.append("Template-based response formatting")
        
        # System status determination
        if retrieval_stats["index_built"] and doc_processor.get_chunk_count() > 0:
            system_status = "operational"
        elif doc_processor.get_chunk_count() > 0:
            system_status = "processing"
        else:
            system_status = "awaiting_documents"
        
        stats = {
            "documents": {
                "total_documents": doc_processor.get_document_count(),
                "total_chunks": doc_processor.get_chunk_count(),
                "last_updated": doc_processor.get_last_update()
            },
            "retrieval": {
                "index_built": retrieval_stats["index_built"],
                "last_index_update": retrieval_stats["last_index_update"],
                "embeddings_cached": retrieval_stats["embeddings_stats"].get("cached_embeddings", 0),
                "keyword_index_terms": retrieval_stats["keyword_stats"].get("unique_terms", 0),
                "keyword_index_status": retrieval_stats["keyword_stats"].get("status", "inactive")
            },
            "table_awareness": {
                "total_tables_detected": table_stats["total_tables_detected"],
                "pages_with_tables": table_stats["pages_with_tables"], 
                "chunks_with_table_content": table_stats["chunks_with_table_content"],
                "documents_with_tables": table_stats["documents_with_tables"]
            },
            "system": {
                "status": system_status,
                "pipeline_components": pipeline_components,
                "advanced_features": advanced_features,
                "capabilities": {
                    "semantic_search": retrieval_stats["embeddings_stats"].get("cached_embeddings", 0) > 0,
                    "keyword_search": retrieval_stats["keyword_stats"].get("status") == "active",
                    "table_processing": table_stats["total_tables_detected"] > 0,
                    "hybrid_retrieval": retrieval_stats["index_built"],
                    "llm_processing": True  # You have Mistral integration
                }
            }
        }
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))