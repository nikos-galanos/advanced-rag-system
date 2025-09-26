"""
Embeddings engine for generating and managing vector representations.
"""
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime

from src.config import settings
from src.utils.helpers import setup_logger
from src.utils.llm_client import MistralClient

logger = setup_logger(__name__)


class EmbeddingsEngine:
    """Handles embedding generation, storage, and similarity search."""
    
    def __init__(self):
        self.llm_client = MistralClient()
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.embedding_dim: Optional[int] = None
        
        # Load existing embeddings if available
        self._load_embeddings_cache()
        
    async def generate_embeddings_for_chunks(self, chunks: List[Any], batch_size: int = 10) -> List[np.ndarray]:
        """
        Generate embeddings for document chunks in batches.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks in batches of {batch_size}")
        
        embeddings = []
        chunks_without_embeddings = []
        
        # Check cache first
        for chunk in chunks:
            cache_key = self._get_cache_key(chunk.text, chunk.id)
            if cache_key in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[cache_key])
                logger.debug(f"Using cached embedding for chunk {chunk.id}")
            else:
                embeddings.append(None)  
                chunks_without_embeddings.append((len(embeddings) - 1, chunk))
        
        # Generate embeddings for uncached chunks
        if chunks_without_embeddings:
            logger.info(f"Generating {len(chunks_without_embeddings)} new embeddings")
            
            # Process in batches
            for i in range(0, len(chunks_without_embeddings), batch_size):
                batch = chunks_without_embeddings[i:i + batch_size]
                batch_texts = [chunk.text for _, chunk in batch]
                
                try:
                    batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                    
                    if batch_embeddings:
                        for j, (embedding_idx, chunk) in enumerate(batch):
                            if j < len(batch_embeddings):
                                embedding_vector = np.array(batch_embeddings[j])
                                embeddings[embedding_idx] = embedding_vector
                                
                                # Cache the embedding
                                cache_key = self._get_cache_key(chunk.text, chunk.id)
                                self.embeddings_cache[cache_key] = embedding_vector
                                
                                # Update chunk object
                                chunk.embedding = embedding_vector
                                
                                # Set embedding dimension if not set
                                if self.embedding_dim is None:
                                    self.embedding_dim = len(embedding_vector)
                    
                    # Small delay between batches to respect rate limits
                    if i + batch_size < len(chunks_without_embeddings):
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {str(e)}")
                    # Fill with zero vectors as fallback
                    for embedding_idx, chunk in batch:
                        if embeddings[embedding_idx] is None:
                            fallback_embedding = self._generate_fallback_embedding(chunk.text)
                            embeddings[embedding_idx] = fallback_embedding
                            chunk.embedding = fallback_embedding
        
        # Save updated cache
        self._save_embeddings_cache()
        
        # Validate all embeddings are generated
        valid_embeddings = []
        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                valid_embeddings.append(embedding)
            else:
                # Final fallback
                fallback_embedding = self._generate_fallback_embedding(chunks[i].text)
                valid_embeddings.append(fallback_embedding)
                chunks[i].embedding = fallback_embedding
        
        logger.info(f"Successfully generated {len(valid_embeddings)} embeddings")
        return valid_embeddings
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        cache_key = self._get_cache_key(query, "query")
        
        if cache_key in self.embeddings_cache:
            logger.debug("Using cached query embedding")
            return self.embeddings_cache[cache_key]
        
        try:
            embeddings = await self._generate_batch_embeddings([query])
            if embeddings and len(embeddings) > 0:
                embedding_vector = np.array(embeddings[0])
                self.embeddings_cache[cache_key] = embedding_vector
                self._save_embeddings_cache()
                return embedding_vector
            else:
                logger.warning("Failed to generate query embedding, using fallback")
                return self._generate_fallback_embedding(query)
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return self._generate_fallback_embedding(query)
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts using Mistral API."""
        try:
            # Clean and prepare texts
            cleaned_texts = [self._prepare_text_for_embedding(text) for text in texts]
            
            # Call Mistral embeddings API
            embeddings = await self.llm_client.generate_embeddings(cleaned_texts)
            
            if embeddings:
                logger.debug(f"Generated {len(embeddings)} embeddings via Mistral API")
                return embeddings
            else:
                logger.warning("Mistral API returned no embeddings")
                return None
                
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            return None
    
    def _prepare_text_for_embedding(self, text: str) -> str:
        """Prepare text for optimal embedding generation."""
        # Truncate if too long (Mistral has token limits)
        max_chars = 8000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        # Clean excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate a simple fallback embedding using TF-IDF-like approach."""
        # Simple word-based embedding as fallback
        words = text.lower().split()
        
        # Use a fixed dimension
        fallback_dim = 1536  # Mistral's embedding dimension
        if self.embedding_dim:
            fallback_dim = self.embedding_dim
        
        # Create a simple hash-based embedding
        embedding = np.zeros(fallback_dim)
        
        for i, word in enumerate(words[:100]):  
            word_hash = hash(word) % fallback_dim
            embedding[word_hash] += 1.0 / (i + 1)  # Position weighting
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        logger.debug("Generated fallback embedding")
        return embedding
    
    def compute_similarity_scores(self, query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarity scores between query and chunks."""
        if not chunk_embeddings:
            return []
        
        try:
            # Convert to numpy array for efficient computation
            chunk_matrix = np.array(chunk_embeddings)
            
            # Normalize vectors
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            chunk_norms = chunk_matrix / (np.linalg.norm(chunk_matrix, axis=1, keepdims=True) + 1e-10)
            
            # Compute cosine similarity
            similarities = np.dot(chunk_norms, query_norm)
            
            # Ensure scores are in [0, 1] range
            similarities = np.clip(similarities, 0.0, 1.0)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error computing similarity scores: {str(e)}")
            # Return zero similarities as fallback
            return [0.0] * len(chunk_embeddings)
    
    def find_most_similar(self, query_embedding: np.ndarray, chunks: List[Any], top_k: int = 10, min_similarity: float = 0.0) -> List[Tuple[Any, float]]:
        """
        Find most similar chunks to query.
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not chunks:
            return []
        
        # Get embeddings for all chunks
        chunk_embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                chunk_embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if not chunk_embeddings:
            logger.warning("No chunks have embeddings")
            return []
        
        # Compute similarities
        similarities = self.compute_similarity_scores(query_embedding, chunk_embeddings)
        
        # Pair chunks with scores and filter by minimum similarity
        scored_chunks = [
            (chunk, score) for chunk, score in zip(valid_chunks, similarities)
            if score >= min_similarity
        ]
        
        # Sort by similarity score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return scored_chunks[:top_k]
    
    def _get_cache_key(self, text: str, identifier: str) -> str:
        """Generate cache key for embedding."""
        import hashlib
        content = f"{text[:100]}_{identifier}"  # Use first 100 chars + ID
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_embeddings_cache(self):
        """Load embeddings cache from disk."""
        cache_file = os.path.join(settings.data_dir, "embeddings_cache.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embeddings_cache = cache_data.get("embeddings", {})
                    self.embedding_dim = cache_data.get("embedding_dim")
                    logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Could not load embeddings cache: {str(e)}")
            self.embeddings_cache = {}
    
    def _save_embeddings_cache(self):
        """Save embeddings cache to disk."""
        cache_file = os.path.join(settings.data_dir, "embeddings_cache.pkl")
        
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data = {
                "embeddings": self.embeddings_cache,
                "embedding_dim": self.embedding_dim,
                "saved_at": datetime.now()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.debug(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings cache."""
        return {
            "cached_embeddings": len(self.embeddings_cache),
            "embedding_dimension": self.embedding_dim,
            "cache_size_mb": len(str(self.embeddings_cache).encode()) / (1024 * 1024)
        }