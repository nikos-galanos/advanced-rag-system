"""
BM25 keyword search implementation for hybrid retrieval.
"""
import math
import pickle
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import re

from src.config import settings
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class BM25KeywordSearch:
    """BM25 algorithm implementation for keyword-based document retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with parameters.
        
        Args:
            k1: Controls term frequency scaling (1.2-2.0)
            b: Controls document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        
        # Index structures
        self.document_frequencies: Dict[str, int] = defaultdict(int)  # df(t)
        self.inverse_document_frequencies: Dict[str, float] = {}  # idf(t)
        self.document_lengths: Dict[str, int] = {}
        self.average_document_length: float = 0.0
        self.total_documents: int = 0
        
        # Term-document matrix
        self.term_document_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Document metadata
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load existing index if available
        self._load_index()
        
    def build_index(self, chunks: List[Any]) -> None:
        """
        Build BM25 index from document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks")
        
        # Reset index
        self.document_frequencies.clear()
        self.term_document_matrix.clear()
        self.document_lengths.clear()
        self.document_metadata.clear()
        
        # Process each chunk
        for chunk in chunks:
            doc_id = chunk.id
            text = chunk.text
            
            # Tokenize and process text
            terms = self._tokenize_and_process(text)
            
            # Store document length
            self.document_lengths[doc_id] = len(terms)
            
            # Store metadata
            self.document_metadata[doc_id] = chunk.metadata
            
            # Count term frequencies in this document
            term_counts = Counter(terms)
            
            # Update term-document matrix and document frequencies
            for term, count in term_counts.items():
                self.term_document_matrix[term][doc_id] = count
                if doc_id not in [doc for doc in self.term_document_matrix[term].keys()]:
                    self.document_frequencies[term] += 1
        
        # Update document frequencies for all terms
        for term in self.term_document_matrix:
            self.document_frequencies[term] = len(self.term_document_matrix[term])
        
        # Calculate statistics
        self.total_documents = len(chunks)
        if self.total_documents > 0:
            self.average_document_length = sum(self.document_lengths.values()) / self.total_documents
        
        # Calculate IDF values
        self._calculate_idf_values()
        
        # Save index
        self._save_index()
        
        logger.info(f"BM25 index built: {len(self.term_document_matrix)} unique terms, "
                   f"{self.total_documents} documents, avg doc length: {self.average_document_length:.1f}")
    
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of (doc_id, bm25_score, metadata) tuples
        """
        if self.total_documents == 0:
            logger.warning("BM25 index is empty")
            return []
        
        # Process query
        query_terms = self._tokenize_and_process(query)
        if not query_terms:
            return []
        
        # Calculate BM25 scores for all documents
        document_scores: Dict[str, float] = defaultdict(float)
        
        for term in query_terms:
            if term in self.term_document_matrix:
                idf_score = self.inverse_document_frequencies.get(term, 0.0)
                
                # Calculate BM25 score for each document containing this term
                for doc_id, term_frequency in self.term_document_matrix[term].items():
                    doc_length = self.document_lengths.get(doc_id, 0)
                    
                    # BM25 formula
                    score_component = idf_score * (
                        term_frequency * (self.k1 + 1) / 
                        (term_frequency + self.k1 * (1 - self.b + self.b * doc_length / self.average_document_length))
                    )
                    
                    document_scores[doc_id] += score_component
        
        # Filter by minimum score and sort
        scored_documents = [
            (doc_id, score, self.document_metadata.get(doc_id, {}))
            for doc_id, score in document_scores.items()
            if score >= min_score
        ]
        
        # Sort by BM25 score (descending)
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"BM25 search for '{query}' returned {len(scored_documents)} results")
        
        return scored_documents[:top_k]
    
    def get_term_statistics(self, term: str) -> Dict[str, Any]:
        """Get statistics for a specific term."""
        if term not in self.term_document_matrix:
            return {"exists": False}
        
        return {
            "exists": True,
            "document_frequency": self.document_frequencies.get(term, 0),
            "inverse_document_frequency": self.inverse_document_frequencies.get(term, 0.0),
            "total_occurrences": sum(self.term_document_matrix[term].values()),
            "documents_containing": len(self.term_document_matrix[term])
        }
    
    def _tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize and process text for indexing.
        
        Args:
            text: Input text
            
        Returns:
            List of processed terms
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
            'they', 'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Remove very short tokens and stopwords
            if len(token) > 2 and token not in stopwords and token.isalpha():
                processed_tokens.append(token)
        
        return processed_tokens
    
    def _calculate_idf_values(self):
        """Calculate IDF (Inverse Document Frequency) values for all terms."""
        self.inverse_document_frequencies.clear()
        
        for term, df in self.document_frequencies.items():
            if df > 0:
                # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                self.inverse_document_frequencies[term] = max(idf, 0.0)  # Ensure non-negative
    
    def _save_index(self):
        """Save BM25 index to disk."""
        index_file = os.path.join(settings.data_dir, "bm25_index.pkl")
        
        try:
            os.makedirs(os.path.dirname(index_file), exist_ok=True)
            
            index_data = {
                "document_frequencies": dict(self.document_frequencies),
                "inverse_document_frequencies": self.inverse_document_frequencies,
                "document_lengths": self.document_lengths,
                "average_document_length": self.average_document_length,
                "total_documents": self.total_documents,
                "term_document_matrix": {
                    term: dict(doc_dict) for term, doc_dict in self.term_document_matrix.items()
                },
                "document_metadata": self.document_metadata,
                "k1": self.k1,
                "b": self.b
            }
            
            with open(index_file, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"Saved BM25 index with {len(self.term_document_matrix)} terms")
            
        except Exception as e:
            logger.error(f"Error saving BM25 index: {str(e)}")
    
    def _load_index(self):
        """Load BM25 index from disk."""
        index_file = os.path.join(settings.data_dir, "bm25_index.pkl")
        
        try:
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.document_frequencies = defaultdict(int, index_data.get("document_frequencies", {}))
                self.inverse_document_frequencies = index_data.get("inverse_document_frequencies", {})
                self.document_lengths = index_data.get("document_lengths", {})
                self.average_document_length = index_data.get("average_document_length", 0.0)
                self.total_documents = index_data.get("total_documents", 0)
                
                # Restore term-document matrix
                tdm_data = index_data.get("term_document_matrix", {})
                self.term_document_matrix = defaultdict(lambda: defaultdict(int))
                for term, doc_dict in tdm_data.items():
                    self.term_document_matrix[term] = defaultdict(int, doc_dict)
                
                self.document_metadata = index_data.get("document_metadata", {})
                self.k1 = index_data.get("k1", self.k1)
                self.b = index_data.get("b", self.b)
                
                logger.info(f"Loaded BM25 index with {len(self.term_document_matrix)} terms, "
                           f"{self.total_documents} documents")
                
        except Exception as e:
            logger.warning(f"Could not load BM25 index: {str(e)}")
            # Initialize empty index
            self.document_frequencies = defaultdict(int)
            self.term_document_matrix = defaultdict(lambda: defaultdict(int))
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        if self.total_documents == 0:
            return {"status": "empty"}
        
        return {
            "status": "active",
            "total_documents": self.total_documents,
            "unique_terms": len(self.term_document_matrix),
            "average_document_length": self.average_document_length,
            "total_term_occurrences": sum(
                sum(doc_dict.values()) for doc_dict in self.term_document_matrix.values()
            ),
            "parameters": {
                "k1": self.k1,
                "b": self.b
            }
        }