"""
Query processing and transformation for the RAG system.
"""
from typing import Dict, Any
from src.utils.helpers import clean_query, setup_logger
from src.utils.llm_client import MistralClient

logger = setup_logger(__name__)


class QueryProcessor:
    """Handles query intent detection and transformation using LLM."""
    
    def __init__(self):
        self.llm_client = MistralClient()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query using advanced LLM-based intent detection."""
        # Clean the query
        cleaned_query = clean_query(query)
        
        # Use LLM for sophisticated intent detection
        intent_analysis = await self.llm_client.detect_intent(cleaned_query)
        
        # Transform the query for better retrieval
        transformed_query = await self._transform_query(cleaned_query, intent_analysis)
        
        processed = {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "intent": intent_analysis.get("intent", "informational"),
            "trigger_search": intent_analysis.get("trigger_search", True),
            "query_type": intent_analysis.get("query_type", "explanation"),
            "confidence": intent_analysis.get("confidence", 0.5),
            "reasoning": intent_analysis.get("reasoning", ""),
            "detection_method": intent_analysis.get("method", "llm"),
            "transformed_query": transformed_query,
            "search_strategy": self._determine_search_strategy(intent_analysis)
        }
        
        logger.info(f"Query processed - Intent: {processed['intent']}, Type: {processed['query_type']}, Search: {processed['trigger_search']}")
        return processed
    
    async def _transform_query(self, query: str, intent_analysis: Dict[str, Any]) -> str:
        """Transform query to improve retrieval based on intent analysis."""
        
        # If high confidence from LLM, apply specific transformations
        if intent_analysis.get("confidence", 0) > 0.8 and intent_analysis.get("method") != "rule-based":
            
            # If it's a greeting, don't transform
            if intent_analysis.get("intent") == "greeting":
                return query
            
            # For questions, we can expand with context
            if intent_analysis.get("intent") == "question":
                query_type = intent_analysis.get("query_type", "explanation")
                
                if query_type == "list":
                    # Add context for list queries
                    if "list" not in query.lower():
                        return f"list of {query.replace('?', '').strip()}"
                
                elif query_type == "comparison":
                    # Add comparison context
                    if "compare" not in query.lower() and "difference" not in query.lower():
                        return f"comparison of {query.replace('?', '').strip()}"
        
        # Basic query cleaning and expansion
        expanded_query = self._expand_query_with_synonyms(query)
        return expanded_query
    
    def _expand_query_with_synonyms(self, query: str) -> str:
        """Basic query expansion with common synonyms."""
        
        # Simple synonym mapping
        synonym_map = {
            "cost": ["price", "expense", "fee"],
            "benefit": ["advantage", "profit", "gain"],
            "issue": ["problem", "challenge", "difficulty"],
            "method": ["approach", "technique", "way"],
            "result": ["outcome", "effect", "consequence"],
            "important": ["significant", "crucial", "key"],
            "different": ["various", "distinct", "separate"],
            "increase": ["grow", "rise", "expand"],
            "decrease": ["reduce", "decline", "drop"]
        }
        
        words = query.lower().split()
        expanded_terms = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip('.,!?')
            if clean_word in synonym_map:
                # Add original word plus first synonym
                expanded_terms.append(word)
                expanded_terms.append(synonym_map[clean_word][0])
            else:
                expanded_terms.append(word)
        
        # If expansion happened, create expanded query
        if len(expanded_terms) > len(words):
            return ' '.join(expanded_terms[:len(words) + 2])  # Limit expansion
        
        return query
    
    def _determine_search_strategy(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the search strategy based on intent analysis."""
        
        query_type = intent_analysis.get("query_type", "explanation")
        intent = intent_analysis.get("intent", "informational")
        
        strategy = {
            "semantic_weight": 0.7,  # Default semantic vs keyword balance
            "keyword_weight": 0.3,
            "rerank_method": "mmr",  # maximal marginal relevance
            "diversity_threshold": 0.3,
            "min_similarity": 0.5
        }
        
        # Adjust strategy based on query type
        if query_type == "factual":
            # For factual queries, prefer precision
            strategy["semantic_weight"] = 0.8
            strategy["min_similarity"] = 0.6
            
        elif query_type == "list":
            # For list queries, prefer diversity
            strategy["diversity_threshold"] = 0.5
            strategy["rerank_method"] = "diversity_mmr"
            
        elif query_type == "summary":
            # For summaries, get diverse content
            strategy["semantic_weight"] = 0.6
            strategy["keyword_weight"] = 0.4
            strategy["diversity_threshold"] = 0.4
            
        elif query_type == "comparison":
            # For comparisons, need diverse perspectives
            strategy["diversity_threshold"] = 0.6
            strategy["rerank_method"] = "comparison_mmr"
        
        # Adjust based on intent
        if intent == "command":
            # Commands often want specific, direct results
            strategy["min_similarity"] = 0.7
            strategy["semantic_weight"] = 0.9
        
        return strategy
    
    def should_refuse_query(self, query: str, intent_analysis: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if query should be refused based on content policies."""
        
        query_lower = query.lower()
        
        # PII detection patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone
        ]
        
        import re
        for pattern in pii_patterns:
            if re.search(pattern, query):
                return True, "Query contains potential personally identifiable information (PII). Please rephrase without sensitive data."
        
        # Medical/legal disclaimer triggers
        medical_legal_terms = [
            'medical advice', 'legal advice', 'diagnosis', 'prescribe', 'treatment',
            'lawsuit', 'attorney', 'lawyer', 'sue', 'court case'
        ]
        
        if any(term in query_lower for term in medical_legal_terms):
            disclaimer = "I cannot provide medical or legal advice. Please consult with qualified professionals for such matters."
            return True, disclaimer
        
        # Harmful content detection
        harmful_terms = [
            'hack', 'exploit', 'illegal', 'fraud', 'steal', 'pirate',
            'violence', 'harm', 'weapon', 'drug dealing'
        ]
        
        if any(term in query_lower for term in harmful_terms):
            return True, "I cannot assist with potentially harmful or illegal activities."
        
        return False, ""