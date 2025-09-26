"""
Query processing and transformation for the RAG system.
"""
from typing import Dict, Any
from src.utils.helpers import clean_query, setup_logger
from src.utils.llm_client import MistralClient
from src.utils.query_transformer import HybridQueryTransformer

logger = setup_logger(__name__)


class QueryProcessor:
    """Handles query intent detection and advanced transformation using hybrid approach."""
    
    def __init__(self):
        self.llm_client = MistralClient()
        self.query_transformer = HybridQueryTransformer(self.llm_client)
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query using advanced LLM-based intent detection and hybrid transformation."""
        # Clean the query
        cleaned_query = clean_query(query)
        
        # Use LLM for sophisticated intent detection
        intent_analysis = await self.llm_client.detect_intent(cleaned_query)
        
        # Advanced hybrid query transformation using the transformer
        transformation_results = await self.query_transformer.transform_query(
            cleaned_query, 
            intent_analysis, 
            context
        )
        
        # Determine advanced search strategy based on transformation results
        search_strategy = self._determine_advanced_search_strategy(intent_analysis, transformation_results)
        
        processed = {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "intent": intent_analysis.get("intent", "informational"),
            "trigger_search": intent_analysis.get("trigger_search", True),
            "query_type": intent_analysis.get("query_type", "explanation"),
            "confidence": intent_analysis.get("confidence", 0.5),
            "reasoning": intent_analysis.get("reasoning", ""),
            "detection_method": intent_analysis.get("method", "rule-based"),
            
            # Enhanced transformation results from HybridQueryTransformer
            "transformed_query": transformation_results["final_query"],
            "synonym_expanded": transformation_results["synonym_expanded"],
            "llm_enhanced": transformation_results["llm_enhanced"],
            "search_variations": transformation_results["search_variations"],
            "query_keywords": transformation_results["query_keywords"],
            "transformation_strategy": transformation_results["transformation_strategy"],
            
            # Advanced search strategy
            "search_strategy": search_strategy
        }
        
        logger.info(f"Query processed - Intent: {processed['intent']}, Type: {processed['query_type']}, "
                   f"Enhanced: {bool(transformation_results['llm_enhanced'])}, "
                   f"Variations: {len(transformation_results['search_variations'])}")
        return processed
    
    def _determine_advanced_search_strategy(self, intent_analysis: Dict[str, Any], transformation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine advanced search strategy based on intent and transformation results."""
        
        query_type = intent_analysis.get("query_type", "explanation")
        intent = intent_analysis.get("intent", "informational")
        has_llm_enhancement = transformation_results.get("llm_enhanced") is not None
        
        strategy = {
            "semantic_weight": 0.7,  # Default semantic vs keyword balance
            "keyword_weight": 0.3,
            "rerank_method": "mmr",  # maximal marginal relevance
            "diversity_threshold": 0.3,
            "min_similarity": 0.5,
            "use_query_variations": len(transformation_results.get("search_variations", [])) > 1,
            "multi_pass_search": False,
            "boost_exact_matches": True
        }
        
        # Adjust strategy based on query type and enhancement quality
        if query_type == "factual":
            # For factual queries, prefer precision and exact matches
            strategy.update({
                "semantic_weight": 0.8,
                "min_similarity": 0.6,
                "boost_exact_matches": True,
                "diversity_threshold": 0.2
            })
            
        elif query_type == "list":
            # For list queries, prefer diversity and comprehensive results
            strategy.update({
                "diversity_threshold": 0.5,
                "rerank_method": "diversity_mmr",
                "multi_pass_search": True,
                "semantic_weight": 0.6,
                "keyword_weight": 0.4
            })
            
        elif query_type == "summary":
            # For summaries, get diverse content with good coverage
            strategy.update({
                "semantic_weight": 0.6,
                "keyword_weight": 0.4,
                "diversity_threshold": 0.4,
                "multi_pass_search": True
            })
            
        elif query_type == "comparison":
            # For comparisons, need diverse perspectives and comprehensive coverage
            strategy.update({
                "diversity_threshold": 0.6,
                "rerank_method": "comparison_mmr",
                "multi_pass_search": True,
                "semantic_weight": 0.65,
                "keyword_weight": 0.35
            })
        
        # Adjust based on LLM enhancement quality
        if has_llm_enhancement:
            # If we have good LLM enhancement, rely more on semantic search
            strategy["semantic_weight"] = min(0.9, strategy["semantic_weight"] + 0.1)
            strategy["keyword_weight"] = 1.0 - strategy["semantic_weight"]
        
        # Adjust based on intent
        if intent == "command":
            # Commands often want specific, direct results
            strategy.update({
                "min_similarity": 0.7,
                "semantic_weight": 0.85,
                "boost_exact_matches": True
            })
        
        return strategy
    
    def should_refuse_query(self, query: str, intent_analysis: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if query should be refused based on enhanced content policies."""
        
        query_lower = query.lower()
        
        # Enhanced PII detection patterns
        import re
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{9}\b'  # Potential SSN without dashes
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, query):
                return True, "Query contains potential personally identifiable information (PII). Please rephrase without sensitive data."
        
        # Enhanced medical/legal disclaimer triggers
        medical_legal_terms = [
            'medical advice', 'legal advice', 'diagnosis', 'prescribe', 'treatment',
            'lawsuit', 'attorney', 'lawyer', 'sue', 'court case', 'medication',
            'dosage', 'symptoms', 'disease', 'illness', 'legal counsel'
        ]
        
        if any(term in query_lower for term in medical_legal_terms):
            disclaimer = "I cannot provide medical or legal advice. Please consult with qualified professionals for such matters."
            return True, disclaimer
        
        # Enhanced harmful content detection
        harmful_terms = [
            'hack', 'exploit', 'illegal', 'fraud', 'steal', 'pirate',
            'violence', 'harm', 'weapon', 'drug dealing', 'money laundering'
        ]
        
        if any(term in query_lower for term in harmful_terms):
            return True, "I cannot assist with potentially harmful or illegal activities."
        
        return False, ""