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
        
        # Step 2: LLM-powered query enhancement (lower threshold for more enhancement)
        llm_enhanced = None
        if intent_analysis.get("confidence", 0) > 0.5 and intent_analysis.get("trigger_search", False):
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
        
        # Add MUCH more comprehensive synonyms
        comprehensive_synonyms = {
            # ML/AI terms
            "machine learning": ["ML", "artificial intelligence", "AI", "deep learning"],
            "learning": ["training", "modeling", "algorithms"],
            "machine": ["computer", "artificial", "automated"],
            
            # News/information terms  
            "news": ["updates", "developments", "advances", "progress", "breakthroughs"],
            "recent": ["latest", "new", "current", "updated"],
            "information": ["data", "details", "facts", "insights"],
            
            # Question terms
            "what": ["which", "describe"],
            "how": ["method", "process", "way"],
            "benefits": ["advantages", "profits", "gains", "value"],
            "issues": ["problems", "challenges", "difficulties"],
            
            # Business terms
            "cost": ["price", "expense", "fee", "budget"],
            "profit": ["revenue", "income", "earnings", "returns"],
            "strategy": ["approach", "plan", "method", "framework"],
            "market": ["industry", "sector", "domain"],
            "customer": ["client", "user", "consumer"],
            
            # Technical terms
            "system": ["platform", "framework", "infrastructure"],
            "performance": ["efficiency", "speed", "optimization"],
            "method": ["approach", "technique", "algorithm", "procedure"],
            "solution": ["fix", "resolution", "answer", "remedy"],
            
            # General enhancement terms
            "increase": ["grow", "rise", "expand", "boost"],
            "decrease": ["reduce", "decline", "drop", "lower"],
            "compare": ["contrast", "evaluate", "analyze"],
            "explain": ["describe", "clarify", "detail"],
            "find": ["locate", "identify", "discover", "search"]
        }
        
        # Merge with domain-specific synonyms
        synonyms.update(comprehensive_synonyms)
        
        # Process each word and add synonyms
        added_synonyms = 0
        max_synonyms = 6  # Limit to prevent over-expansion
        
        for word in words:
            clean_word = word.strip('.,!?()[]')
            expanded_terms.append(word)
            
            # Check for exact matches
            if clean_word in synonyms and added_synonyms < max_synonyms:
                if query_type == "factual":
                    # For factual queries, add 1 precise synonym
                    expanded_terms.append(synonyms[clean_word][0])
                    added_synonyms += 1
                elif query_type in ["explanation", "list"]:
                    # For explanations and lists, add 2 contextual synonyms
                    expanded_terms.extend(synonyms[clean_word][:2])
                    added_synonyms += 2
                else:
                    # Default: add 1 synonym
                    expanded_terms.append(synonyms[clean_word][0])
                    added_synonyms += 1
            
            # Also check for multi-word phrases
            if added_synonyms < max_synonyms:
                for phrase, phrase_synonyms in synonyms.items():
                    if ' ' in phrase and phrase in query.lower() and added_synonyms < max_synonyms:
                        expanded_terms.extend(phrase_synonyms[:1])
                        added_synonyms += 1
                        break
        
        expanded_query = ' '.join(expanded_terms)
        
        # If no synonyms were added, try alternative approach
        if len(expanded_terms) <= len(words):
            # Add domain-specific terms based on detected context
            if domain == "technical":
                expanded_query += " technology system implementation"
            elif domain == "business":
                expanded_query += " business strategy market"
            elif "machine learning" in query.lower():
                expanded_query += " AI algorithms data science"
            
        return expanded_query.strip()
    
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
        if llm_enhanced and confidence > 0.6:  # Lowered threshold
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
        query_lower = query.lower()
        
        # Generate more comprehensive variations
        if query_type == "factual":
            variations.extend([
                query.replace("what is", "definition of").replace("what are", "examples of"),
                query.replace("?", "").strip(),
                f"explain {query.replace('what is ', '').replace('what are ', '').replace('?', '')}"
            ])
            
        elif query_type == "list":
            base_query = query.replace("list", "").replace("what are some", "").strip()
            variations.extend([
                f"list of {base_query}",
                f"examples of {base_query}",
                f"types of {base_query}",
                query.replace("list", "enumerate").replace("what are some", "show me")
            ])
            
        elif query_type == "comparison":
            variations.extend([
                query.replace("compare", "difference between"),
                query.replace("compare", "similarities and differences"),
                f"analysis of {query.replace('compare ', '').replace('comparison of ', '')}"
            ])
        
        # Add domain-specific variations
        if "machine learning" in query_lower or "ml" in query_lower:
            variations.extend([
                query.replace("machine learning", "ML").replace("artificial intelligence", "AI"),
                query.replace("ML", "artificial intelligence"),
                query.replace("news", "updates").replace("developments", "advances")
            ])
        
        # Add question format variations
        if not query.endswith("?"):
            variations.append(f"{query}?")
        
        if "what are" in query_lower:
            variations.append(query.replace("what are", "show me").replace("What are", "Show me"))
        
        if "news" in query_lower:
            variations.extend([
                query.replace("news", "latest developments"),
                query.replace("news", "recent updates"),
                query.replace("news", "current trends")
            ])
        
        # Remove duplicates and empty strings, preserve order
        seen = set()
        unique_variations = []
        for v in variations:
            clean_v = v.strip()
            if clean_v and clean_v not in seen and len(clean_v) > 3:
                seen.add(clean_v)
                unique_variations.append(clean_v)
        
        return unique_variations[:6]  # Limit to 6 variations for efficiency
    
    def _detect_domain_context(self, query: str) -> str:
        """Detect the domain context of the query."""
        
        query_lower = query.lower()
        
        business_terms = ["cost", "profit", "revenue", "market", "strategy", "business", "company", "customer"]
        technical_terms = ["system", "method", "algorithm", "performance", "implementation", "code", "software", "machine learning", "AI"]
        academic_terms = ["research", "study", "analysis", "theory", "hypothesis", "paper", "journal"]
        
        business_count = sum(1 for term in business_terms if term in query_lower)
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        academic_count = sum(1 for term in academic_terms if term in query_lower)
        
        if technical_count >= business_count and technical_count >= academic_count:
            return "technical"
        elif business_count >= academic_count:
            return "business"
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
            "confidence_threshold": 0.5,  # Lowered threshold
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
        return list(dict.fromkeys(keywords))