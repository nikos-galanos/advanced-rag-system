"""
Enhanced query transformation with hybrid LLM + rule-based approach.
"""
from typing import Dict, Any, List
import json
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class HybridQueryTransformer:
    """Combines rule-based and LLM-powered query enhancement."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
        # Enhanced synonym mappings by domain
        self.domain_synonyms = {
            "business": {
                "cost": ["price", "expense", "fee", "budget", "expenditure"],
                "profit": ["revenue", "income", "earnings", "returns", "gains"],
                "strategy": ["approach", "plan", "method", "framework", "roadmap"],
                "market": ["industry", "sector", "domain", "space", "vertical"],
                "customer": ["client", "user", "consumer", "buyer", "patron"]
            },
            "technical": {
                "method": ["approach", "technique", "algorithm", "procedure", "process"],
                "system": ["platform", "framework", "infrastructure", "architecture"],
                "performance": ["efficiency", "speed", "throughput", "optimization"],
                "issue": ["problem", "bug", "error", "challenge", "difficulty"],
                "solution": ["fix", "resolution", "answer", "remedy", "approach"]
            },
            "academic": {
                "research": ["study", "analysis", "investigation", "examination"],
                "result": ["outcome", "finding", "conclusion", "output", "effect"],
                "important": ["significant", "crucial", "key", "critical", "vital"],
                "different": ["distinct", "various", "separate", "diverse", "alternative"]
            }
        }
    
    async def transform_query(self, original_query: str, intent_analysis: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transform query using hybrid approach: synonyms + LLM enhancement.
        
        Args:
            original_query: The original user query
            intent_analysis: Results from intent detection
            context: Optional context about available documents
            
        Returns:
            Dict with transformation results and strategies
        """
        
        # Step 1: Rule-based synonym expansion
        synonym_expanded = self._expand_with_synonyms(original_query, intent_analysis)
        
        # Step 2: LLM-powered query enhancement (if confidence is high enough)
        llm_enhanced = None
        if intent_analysis.get("confidence", 0) > 0.7 and intent_analysis.get("trigger_search", False):
            llm_enhanced = await self._llm_enhance_query(original_query, intent_analysis, context)
        
        # Step 3: Combine approaches intelligently
        final_query = self._combine_transformations(original_query, synonym_expanded, llm_enhanced, intent_analysis)
        
        # Step 4: Generate search variations for better retrieval
        search_variations = self._generate_search_variations(final_query, intent_analysis)
        
        return {
            "original_query": original_query,
            "synonym_expanded": synonym_expanded,
            "llm_enhanced": llm_enhanced,
            "final_query": final_query,
            "search_variations": search_variations,
            "transformation_strategy": self._get_transformation_strategy(intent_analysis),
            "query_keywords": self._extract_enhanced_keywords(final_query)
        }
    
    def _expand_with_synonyms(self, query: str, intent_analysis: Dict[str, Any]) -> str:
        """Rule-based synonym expansion with domain awareness."""
        
        query_type = intent_analysis.get("query_type", "explanation")
        words = query.lower().split()
        expanded_terms = []
        
        # Detect domain context
        domain = self._detect_domain_context(query)
        synonyms = self.domain_synonyms.get(domain, {})
        
        # Add general synonyms to domain-specific ones
        general_synonyms = {
            "increase": ["grow", "rise", "expand", "enhance", "boost"],
            "decrease": ["reduce", "decline", "drop", "lower", "minimize"],
            "compare": ["contrast", "evaluate", "analyze", "assess"],
            "explain": ["describe", "clarify", "elaborate", "detail"],
            "find": ["locate", "identify", "discover", "search", "retrieve"]
        }
        synonyms.update(general_synonyms)
        
        for word in words:
            clean_word = word.strip('.,!?()[]')
            expanded_terms.append(word)
            
            # Add synonyms based on query type
            if clean_word in synonyms:
                if query_type == "factual":
                    # For factual queries, add 1 precise synonym
                    expanded_terms.append(synonyms[clean_word][0])
                elif query_type == "explanation":
                    # For explanations, add 2 contextual synonyms
                    expanded_terms.extend(synonyms[clean_word][:2])
                elif query_type == "list":
                    # For lists, add variation terms
                    expanded_terms.extend(synonyms[clean_word][:1])
        
        # Limit expansion to avoid over-complication
        max_length = len(words) + min(5, len(words))
        return ' '.join(expanded_terms[:max_length])
    
    async def _llm_enhance_query(self, query: str, intent_analysis: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Use LLM to enhance query with context and domain knowledge."""
        
        # Build context-aware prompt
        available_docs = context.get("document_count", 0) if context else 0
        query_type = intent_analysis.get("query_type", "explanation")
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a query enhancement specialist for a document retrieval system.

Your task: Transform the user's query to improve document retrieval while preserving the original intent.

Guidelines:
1. Add relevant domain terms and context
2. Include alternative phrasings that might appear in documents
3. For {query_type} queries, optimize for that specific type
4. Keep the enhanced query concise but comprehensive
5. Don't change the core meaning

Available context: {available_docs} documents in the knowledge base.

Respond with ONLY the enhanced query, no explanations."""
            },
            {
                "role": "user", 
                "content": f"Original query: \"{query}\"\nQuery type: {query_type}\nIntent: {intent_analysis.get('intent')}\n\nEnhanced query:"
            }
        ]
        
        try:
            enhanced = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=100
            )
            
            if enhanced and len(enhanced.strip()) > len(query.strip()):
                logger.info(f"LLM enhanced query: '{query}' -> '{enhanced.strip()}'")
                return enhanced.strip()
            else:
                logger.info("LLM enhancement didn't improve query significantly")
                return None
                
        except Exception as e:
            logger.warning(f"LLM query enhancement failed: {str(e)}")
            return None
    
    def _combine_transformations(self, original: str, synonym_expanded: str, llm_enhanced: str, intent_analysis: Dict[str, Any]) -> str:
        """Intelligently combine different transformation approaches."""
        
        query_type = intent_analysis.get("query_type", "explanation")
        confidence = intent_analysis.get("confidence", 0.5)
        
        # Decision logic for combination
        if llm_enhanced and confidence > 0.8:
            # High confidence: prefer LLM enhancement
            if query_type in ["explanation", "comparison"]:
                # For complex queries, blend LLM with key synonyms
                key_synonyms = self._extract_key_synonyms(original, synonym_expanded)
                return f"{llm_enhanced} {key_synonyms}".strip()
            else:
                # For simple queries, use LLM enhancement
                return llm_enhanced
                
        elif synonym_expanded and len(synonym_expanded.split()) > len(original.split()):
            # Medium confidence: prefer synonym expansion
            return synonym_expanded
            
        else:
            # Low confidence: use original with minimal enhancement
            return original
    
    def _generate_search_variations(self, query: str, intent_analysis: Dict[str, Any]) -> List[str]:
        """Generate query variations for multi-pass retrieval."""
        
        variations = [query]  # Always include the main query
        query_type = intent_analysis.get("query_type", "explanation")
        
        if query_type == "factual":
            # For factual queries, create direct variations
            variations.extend([
                query.replace("what is", "definition of"),
                query.replace("?", ""),
                f"explain {query.replace('what is ', '').replace('?', '')}"
            ])
            
        elif query_type == "list":
            # For list queries, add enumeration terms
            variations.extend([
                f"list of {query.replace('list', '').strip()}",
                f"examples of {query.replace('list', '').strip()}",
                query.replace("list", "enumerate")
            ])
            
        elif query_type == "comparison":
            # For comparison queries, add analysis terms
            variations.extend([
                query.replace("compare", "difference between"),
                query.replace("compare", "similarities and differences"),
                f"analysis of {query.replace('compare ', '').replace('comparison of ', '')}"
            ])
        
        # Remove duplicates and empty strings
        variations = list(dict.fromkeys([v.strip() for v in variations if v.strip()]))
        return variations[:4]  # Limit to 4 variations
    
    def _detect_domain_context(self, query: str) -> str:
        """Detect the domain context of the query."""
        
        query_lower = query.lower()
        
        business_terms = ["cost", "profit", "revenue", "market", "strategy", "business", "company", "customer"]
        technical_terms = ["system", "method", "algorithm", "performance", "implementation", "code", "software"]
        academic_terms = ["research", "study", "analysis", "theory", "hypothesis", "paper", "journal"]
        
        business_count = sum(1 for term in business_terms if term in query_lower)
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        academic_count = sum(1 for term in academic_terms if term in query_lower)
        
        if business_count >= technical_count and business_count >= academic_count:
            return "business"
        elif technical_count >= academic_count:
            return "technical"
        elif academic_count > 0:
            return "academic"
        else:
            return "general"
    
    def _extract_key_synonyms(self, original: str, synonym_expanded: str) -> str:
        """Extract the most important synonyms from expansion."""
        
        original_words = set(original.lower().split())
        expanded_words = synonym_expanded.lower().split()
        
        # Get new words that weren't in original
        new_words = [word for word in expanded_words if word not in original_words and len(word) > 2]
        
        # Return top 3 most relevant new terms
        return ' '.join(new_words[:3])
    
    def _get_transformation_strategy(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return the transformation strategy used."""
        
        return {
            "approach": "hybrid",
            "methods_used": ["synonym_expansion", "llm_enhancement", "variation_generation"],
            "query_type_optimization": intent_analysis.get("query_type"),
            "confidence_threshold": 0.7,
            "domain_aware": True
        }
    
    def _extract_enhanced_keywords(self, query: str) -> List[str]:
        """Extract keywords from the enhanced query for search optimization."""
        
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those'
        }
        
        words = [word.strip('.,!?()[]').lower() for word in query.split()]
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Return unique keywords, preserving order
        return list(dict.fromkeys(keywords))# Copy the HybridQueryTransformer content here
