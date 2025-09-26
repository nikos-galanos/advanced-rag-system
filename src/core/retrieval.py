"""
Advanced retrieval engine with hybrid semantic and keyword search.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import asyncio

from src.core.embeddings import EmbeddingsEngine
from src.core.keyword_search import BM25KeywordSearch
from src.core.ingestion import DocumentProcessor
from src.utils.helpers import setup_logger
from src.config import settings

logger = setup_logger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with all scoring information."""
    chunk_id: str
    chunk: Any
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank_position: int
    metadata: Dict[str, Any]


class HybridRetrievalEngine:
    """Advanced retrieval engine combining semantic and keyword search with intelligent re-ranking."""
    
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.embeddings_engine = EmbeddingsEngine()
        self.keyword_search = BM25KeywordSearch()
        
        self.index_built = False
        self.last_index_update: Optional[datetime] = None
        
        # Initialize indices if documents exist
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize indices if documents are already available."""
        chunks = self.document_processor.get_chunks()
        if chunks:
            logger.info(f"Found {len(chunks)} existing chunks, checking if indices need building")
            # Check if indices exist
            keyword_stats = self.keyword_search.get_index_statistics()
            embeddings_stats = self.embeddings_engine.get_cache_stats()
            
            if keyword_stats.get("status") == "active" and embeddings_stats.get("cached_embeddings", 0) > 0:
                self.index_built = True
                logger.info("Existing indices found and loaded")
            else:
                logger.info("Indices need to be built")
    
    async def build_indices(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build or rebuild search indices.
        
        Args:
            force_rebuild: Whether to rebuild even if indices exist
            
        Returns:
            Statistics about the indexing process
        """
        chunks = self.document_processor.get_chunks()
        
        if not chunks:
            logger.warning("No chunks available for indexing")
            return {"status": "no_chunks", "chunks_processed": 0}
        
        if self.index_built and not force_rebuild:
            logger.info("Indices already built and force_rebuild=False")
            return {"status": "already_built", "chunks_processed": len(chunks)}
        
        logger.info(f"Building indices for {len(chunks)} chunks")
        start_time = datetime.now()
        
        # Build keyword search index
        logger.info("Building BM25 keyword search index...")
        self.keyword_search.build_index(chunks)
        
        # Generate embeddings for chunks that don't have them
        logger.info("Generating embeddings for chunks...")
        chunks_without_embeddings = [chunk for chunk in chunks if chunk.embedding is None]
        
        if chunks_without_embeddings:
            await self.embeddings_engine.generate_embeddings_for_chunks(chunks_without_embeddings)
            
            # Save updated chunks with embeddings
            self.document_processor._save_data()
        
        # Update status
        self.index_built = True
        self.last_index_update = datetime.now()
        
        build_time = (datetime.now() - start_time).total_seconds()
        
        stats = {
            "status": "success",
            "chunks_processed": len(chunks),
            "new_embeddings_generated": len(chunks_without_embeddings),
            "build_time_seconds": build_time,
            "index_statistics": {
                "embeddings": self.embeddings_engine.get_cache_stats(),
                "keyword_search": self.keyword_search.get_index_statistics()
            }
        }
        
        logger.info(f"Indices built successfully in {build_time:.1f}s")
        return stats
    
    async def retrieve(self, processed_query: Dict[str, Any], top_k: int = 10) -> List[SearchResult]:
        """
        Perform hybrid search using both semantic and keyword approaches.
        
        Args:
            processed_query: Processed query from QueryProcessor
            top_k: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.index_built:
            logger.info("Indices not built, building now...")
            await self.build_indices()
        
        if not self.index_built:
            logger.error("Could not build indices")
            return []
        
        chunks = self.document_processor.get_chunks()
        if not chunks:
            logger.warning("No chunks available for retrieval")
            return []
        
        # Get search strategy
        search_strategy = processed_query.get("search_strategy", {})
        semantic_weight = search_strategy.get("semantic_weight", 0.7)
        keyword_weight = search_strategy.get("keyword_weight", 0.3)
        min_similarity = search_strategy.get("min_similarity", 0.3)
        use_variations = search_strategy.get("use_query_variations", False)
        
        # Prepare queries for search
        main_query = processed_query.get("transformed_query", processed_query.get("cleaned_query", ""))
        search_queries = [main_query]
        
        if use_variations:
            variations = processed_query.get("search_variations", [])
            search_queries.extend(variations[:3])  # Limit variations
        
        logger.info(f"Performing hybrid search with {len(search_queries)} query variations")
        
        # Collect results from all queries
        all_semantic_results = []
        all_keyword_results = []
        
        for i, query in enumerate(search_queries):
            weight_multiplier = 1.0 - (i * 0.1)  # Reduce weight for variations
            
            # Semantic search
            semantic_results = await self._semantic_search(query, chunks, top_k * 2, min_similarity)
            for chunk, score in semantic_results:
                all_semantic_results.append((chunk, score * weight_multiplier))
            
            # Keyword search
            keyword_results = self._keyword_search(query, top_k * 2)
            for doc_id, score, metadata in keyword_results:
                # Find the chunk by ID
                chunk = next((c for c in chunks if c.id == doc_id), None)
                if chunk:
                    all_keyword_results.append((chunk, score * weight_multiplier))
        
        # Merge and re-rank results
        final_results = self._merge_and_rerank_results(
            all_semantic_results,
            all_keyword_results,
            semantic_weight,
            keyword_weight,
            search_strategy,
            top_k
        )
        
        logger.info(f"Retrieved {len(final_results)} results for query: {main_query}")
        return final_results
    
    async def _semantic_search(self, query: str, chunks: List[Any], top_k: int, min_similarity: float) -> List[Tuple[Any, float]]:
        """Perform semantic search using embeddings."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings_engine.generate_query_embedding(query)
            
            # Find most similar chunks
            similar_chunks = self.embeddings_engine.find_most_similar(
                query_embedding, chunks, top_k, min_similarity
            )
            
            logger.debug(f"Semantic search found {len(similar_chunks)} results for: {query}")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform keyword search using BM25."""
        try:
            results = self.keyword_search.search(query, top_k, min_score=0.1)
            logger.debug(f"Keyword search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _merge_and_rerank_results(
        self,
        semantic_results: List[Tuple[Any, float]],
        keyword_results: List[Tuple[Any, float]], 
        semantic_weight: float,
        keyword_weight: float,
        search_strategy: Dict[str, Any],
        top_k: int
    ) -> List[SearchResult]:
        """Merge semantic and keyword results with intelligent re-ranking."""
        
        # Create a dictionary to combine scores for same chunks
        chunk_scores = {}
        
        # Add semantic scores
        for chunk, semantic_score in semantic_results:
            if chunk.id not in chunk_scores:
                chunk_scores[chunk.id] = {
                    "chunk": chunk,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0
                }
            chunk_scores[chunk.id]["semantic_score"] = max(
                chunk_scores[chunk.id]["semantic_score"], semantic_score
            )
        
        # Add keyword scores
        for chunk, keyword_score in keyword_results:
            if chunk.id not in chunk_scores:
                chunk_scores[chunk.id] = {
                    "chunk": chunk,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0
                }
            chunk_scores[chunk.id]["keyword_score"] = max(
                chunk_scores[chunk.id]["keyword_score"], keyword_score
            )
        
        # Calculate combined scores
        search_results = []
        for chunk_id, scores in chunk_scores.items():
            combined_score = (
                scores["semantic_score"] * semantic_weight + 
                scores["keyword_score"] * keyword_weight
            )
            
            result = SearchResult(
                chunk_id=chunk_id,
                chunk=scores["chunk"],
                semantic_score=scores["semantic_score"],
                keyword_score=scores["keyword_score"],
                combined_score=combined_score,
                rank_position=0,  # Will be set after sorting
                metadata={
                    "semantic_weight": semantic_weight,
                    "keyword_weight": keyword_weight,
                    "search_method": "hybrid"
                }
            )
            search_results.append(result)
        
        # Sort by combined score
        search_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply re-ranking strategy
        reranked_results = self._apply_reranking_strategy(search_results, search_strategy)
        
        # Set rank positions and limit results
        final_results = reranked_results[:top_k]
        for i, result in enumerate(final_results):
            result.rank_position = i + 1
        
        return final_results
    
    def _apply_reranking_strategy(self, results: List[SearchResult], strategy: Dict[str, Any]) -> List[SearchResult]:
        """Apply advanced re-ranking based on search strategy."""
        
        rerank_method = strategy.get("rerank_method", "mmr")
        diversity_threshold = strategy.get("diversity_threshold", 0.3)
        
        if rerank_method == "mmr" or rerank_method == "diversity_mmr":
            # Maximal Marginal Relevance
            return self._apply_mmr_reranking(results, diversity_threshold)
        elif rerank_method == "comparison_mmr":
            # Special MMR for comparison queries
            return self._apply_comparison_mmr(results, diversity_threshold)
        else:
            # Default: just return sorted by combined score
            return results
    
    def _apply_mmr_reranking(self, results: List[SearchResult], diversity_threshold: float) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance re-ranking."""
        
        if len(results) <= 1:
            return results
        
        # Start with the highest scored result
        reranked = [results[0]]
        remaining = results[1:]
        
        while remaining and len(reranked) < len(results):
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.combined_score
                
                # Diversity score (how different from already selected)
                diversity = self._calculate_diversity_score(candidate, reranked)
                
                # MMR score
                mmr_score = relevance - diversity_threshold * (1 - diversity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best candidate to reranked list
            reranked.append(remaining.pop(best_idx))
        
        return reranked
    
    def _calculate_diversity_score(self, candidate: SearchResult, selected: List[SearchResult]) -> float:
        """Calculate diversity score for MMR re-ranking."""
        
        if not selected:
            return 1.0
        
        # Simple diversity based on text similarity
        candidate_text = candidate.chunk.text.lower()
        candidate_words = set(candidate_text.split())
        
        min_diversity = 1.0
        for selected_result in selected:
            selected_text = selected_result.chunk.text.lower()
            selected_words = set(selected_text.split())
            
            # Jaccard similarity
            intersection = len(candidate_words.intersection(selected_words))
            union = len(candidate_words.union(selected_words))
            
            if union > 0:
                similarity = intersection / union
                diversity = 1.0 - similarity
                min_diversity = min(min_diversity, diversity)
        
        return min_diversity
    
    def _apply_comparison_mmr(self, results: List[SearchResult], diversity_threshold: float) -> List[SearchResult]:
        """Apply MMR specifically optimized for comparison queries."""
        
        # For comparison queries, we want diverse perspectives
        # Increase diversity requirement
        enhanced_diversity_threshold = min(0.8, diversity_threshold + 0.2)
        
        return self._apply_mmr_reranking(results, enhanced_diversity_threshold)
    
    def get_index_size(self) -> int:
        """Get the size of the search index."""
        if not self.index_built:
            return 0
        
        keyword_stats = self.keyword_search.get_index_statistics()
        return keyword_stats.get("total_documents", 0)
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval system statistics."""
        
        return {
            "index_built": self.index_built,
            "last_index_update": self.last_index_update,
            "total_chunks": len(self.document_processor.get_chunks()),
            "embeddings_stats": self.embeddings_engine.get_cache_stats(),
            "keyword_stats": self.keyword_search.get_index_statistics()
        }