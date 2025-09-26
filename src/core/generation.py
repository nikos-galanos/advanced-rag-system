"""
Advanced generation engine for creating contextual responses using Mistral AI.
"""
from typing import List, Dict, Any
from src.utils.helpers import setup_logger
from src.utils.llm_client import MistralClient
from src.config import settings
import re

logger = setup_logger(__name__)


class GenerationEngine:
    """Handles answer generation using LLM with retrieved context and advanced features."""
    
    def __init__(self):
        self.llm_client = MistralClient()
    
    async def generate_answer(self, query: Dict[str, Any], search_results: List[Any], include_metadata: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive answer based on query and retrieved chunks.
        
        Args:
            query: Processed query information from QueryProcessor
            search_results: List of SearchResult objects from retrieval
            include_metadata: Whether to include generation metadata
            
        Returns:
            Dict containing answer, citations, and metadata
        """
        logger.info(f"Generating answer for query type: {query.get('query_type')} with {len(search_results)} chunks")
        
        # Check evidence threshold first
        if not self._meets_evidence_threshold(search_results, query):
            return {
                "answer": "Insufficient evidence. I couldn't find enough relevant information in the documents to provide a confident answer to your question.",
                "citations": [],
                "metadata": {
                    "evidence_status": "insufficient",
                    "reason": "Top chunks don't meet similarity threshold",
                    "top_score": max([r.combined_score for r in search_results]) if search_results else 0,
                    "threshold": settings.similarity_threshold
                }
            }
        
        # Select appropriate prompt template based on query type and intent
        template = self._select_prompt_template(query, search_results)
        
        # Build context from retrieved chunks
        context = self._build_context_from_chunks(search_results)
        
        # Generate answer using LLM
        generated_answer = await self._call_llm_for_generation(template, query, context)
        
        if not generated_answer:
            return {
                "answer": "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question.",
                "citations": [],
                "metadata": {"error": "LLM generation failed"}
            }
        
        # Apply hallucination filter
        filtered_answer, hallucination_info = self._apply_hallucination_filter(generated_answer, search_results)
        
        # Generate citations
        citations = self._generate_citations(search_results)
        
        # Build metadata
        metadata = {}
        if include_metadata:
            metadata = {
                "chunks_used": len(search_results),
                "confidence": self._calculate_answer_confidence(search_results),
                "template_used": template["name"],
                "hallucination_check": hallucination_info,
                "citation_count": len(citations),
                "evidence_quality": "sufficient"
            }
        
        return {
            "answer": filtered_answer,
            "citations": citations,
            "metadata": metadata
        }
    
    def _meets_evidence_threshold(self, search_results: List[Any], query: Dict[str, Any]) -> bool:
        """Check if retrieved chunks meet minimum evidence threshold."""
        
        if not search_results:
            return False
        
        # Get search strategy threshold
        search_strategy = query.get("search_strategy", {})
        min_similarity = search_strategy.get("min_similarity", settings.similarity_threshold)
        
        # Check if top result meets threshold
        top_result = search_results[0]
        meets_threshold = top_result.combined_score >= min_similarity
        
        logger.debug(f"Evidence check - Top score: {top_result.combined_score:.3f}, Threshold: {min_similarity:.3f}, Meets: {meets_threshold}")
        
        return meets_threshold
    
    def _select_prompt_template(self, query: Dict[str, Any], search_results: List[Any]) -> Dict[str, str]:
        """Select appropriate prompt template based on query type and intent."""
        
        query_type = query.get("query_type", "explanation")
        intent = query.get("intent", "informational")
        
        templates = {
            "factual": {
                "name": "factual_template",
                "system": "You are a precise AI assistant. Provide direct, factual answers based strictly on the provided context. Be concise and accurate.",
                "format": "Based on the provided documents, {query}\n\nAnswer directly and cite your sources."
            },
            "list": {
                "name": "list_template", 
                "system": "You are an AI assistant that creates well-structured lists. Organize information clearly with bullet points or numbered lists.",
                "format": "Based on the provided documents, {query}\n\nProvide a well-organized list with clear explanations for each item."
            },
            "comparison": {
                "name": "comparison_template",
                "system": "You are an AI assistant that excels at comparative analysis. Present balanced comparisons with clear distinctions.",
                "format": "Based on the provided documents, {query}\n\nProvide a balanced comparison highlighting key similarities and differences."
            },
            "explanation": {
                "name": "explanation_template",
                "system": "You are a knowledgeable AI assistant that provides comprehensive explanations. Make complex topics accessible while maintaining accuracy.",
                "format": "Based on the provided documents, {query}\n\nProvide a comprehensive explanation that covers the key aspects of this topic."
            }
        }
        
        # Select template based on query type, with fallback to explanation
        selected_template = templates.get(query_type, templates["explanation"])
        
        logger.debug(f"Selected template: {selected_template['name']} for query type: {query_type}")
        return selected_template
    
    def _build_context_from_chunks(self, search_results: List[Any]) -> str:
        """Build formatted context string from retrieved chunks."""
        
        context_parts = []
        
        for i, result in enumerate(search_results[:8]):  # Limit to top 8 chunks
            chunk = result.chunk
            
            # Add chunk with metadata for context
            context_part = f"[Source {i+1}: {chunk.metadata.get('document_name', 'Unknown')}]:\n{chunk.text.strip()}\n"
            context_parts.append(context_part)
        
        context = "\n---\n".join(context_parts)
        
        max_context_chars = 6000  # limit for context
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n...[Content truncated for length]"
        
        return context
    
    async def _call_llm_for_generation(self, template: Dict[str, str], query: Dict[str, Any], context: str) -> str:
        """Call LLM to generate answer using selected template."""
        
        user_query = query.get("original_query", "")
        
        messages = [
            {
                "role": "system",
                "content": template["system"] + "\n\nIMPORTANT: Base your answer ONLY on the provided context. If the context doesn't contain enough information, say so. Always cite the sources you use."
            },
            {
                "role": "user",
                "content": f"Context from documents:\n{context}\n\n{template['format'].format(query=user_query)}"
            }
        ]
        
        try:
            answer = await self.llm_client.chat_completion(
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            
            if answer:
                logger.debug(f"Generated answer length: {len(answer)} characters")
                return answer.strip()
            else:
                logger.warning("LLM returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return None
    
    def _apply_hallucination_filter(self, answer: str, search_results: List[Any]) -> tuple[str, Dict[str, Any]]:
        """Apply post-hoc hallucination detection filter with improved support for structured data."""
        
        # Extract key facts from the answer
        answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        
        # Get all source text for verification
        source_texts = [result.chunk.text.lower() for result in search_results]
        combined_sources = " ".join(source_texts)
        
        # Check each sentence for support in sources
        supported_sentences = []
        unsupported_sentences = []
        
        for sentence in answer_sentences:
            if len(sentence) < 10:  # Skip very short sentences
                supported_sentences.append(sentence)
                continue
                
            # Enhanced verification for structured data
            sentence_lower = sentence.lower()
            
            # Extract key entities from sentence
            key_entities = self._extract_key_entities(sentence_lower)
            
            # Check if sentence is supported by sources
            is_supported = self._check_sentence_support(sentence_lower, combined_sources, key_entities)
            
            if is_supported:
                supported_sentences.append(sentence)
            else:
                # Additional check for tabular/structured data
                if self._is_structured_data_sentence(sentence_lower, search_results):
                    logger.info(f"Allowing structured data sentence: {sentence[:50]}...")
                    supported_sentences.append(sentence)
                else:
                    unsupported_sentences.append(sentence)
        
        # Reconstruct answer with only supported sentences
        filtered_answer = ". ".join(supported_sentences)
        if filtered_answer and not filtered_answer.endswith('.'):
            filtered_answer += "."
        
        hallucination_info = {
            "total_sentences": len(answer_sentences),
            "supported_sentences": len(supported_sentences),
            "unsupported_sentences": len(unsupported_sentences),
            "filter_applied": len(unsupported_sentences) > 0,
            "structured_data_detected": any(r.chunk.metadata.get("contains_structured_data", False) for r in search_results)
        }
        
        if len(unsupported_sentences) > 0:
            logger.info(f"Hallucination filter removed {len(unsupported_sentences)} potentially unsupported sentences")
        
        return filtered_answer, hallucination_info
    
    def _extract_key_entities(self, sentence: str) -> List[str]:
        """Extract key entities like codes, dates, names, numbers from sentence."""
        import re
        
        entities = []
        
        # Alphanumeric codes
        code_pattern = r'\b[A-Z]{2,4}\d{1,4}\b'
        entities.extend(re.findall(code_pattern, sentence.upper()))
        
        # Three-letter codes
        three_letter_pattern = r'\b[A-Z]{3}\b'
        entities.extend(re.findall(three_letter_pattern, sentence.upper()))
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # 2025-09-23
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # 9/23/2025
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',   # 9-23-2025
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b'  # 23 Sep 2025
        ]
        for pattern in date_patterns:
            entities.extend(re.findall(pattern, sentence, re.IGNORECASE))
        
        # Numbers
        number_patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # Money: $1,234.56
            r'\d+%',  # Percentages: 25%
            r'\b\d{1,3}(?:,\d{3})*\b'  # Large numbers: 1,234
        ]
        for pattern in number_patterns:
            entities.extend(re.findall(pattern, sentence))
        
        # Names (capitalized words - common proper nouns)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, sentence)
        # Filter out common words that might be capitalized
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But'}
        entities.extend([name for name in potential_names if name not in common_words])
        
        return entities
    
    def _check_sentence_support(self, sentence: str, combined_sources: str, key_entities: List[str]) -> bool:
        """Enhanced support checking with entity matching for any type of content."""
        
        # Method 1: Original word-based approach (relaxed threshold)
        key_words = [word for word in sentence.split() if len(word) > 3 and word.isalpha()]
        if key_words:
            word_support_ratio = sum(1 for word in key_words if word in combined_sources) / len(key_words)
            if word_support_ratio >= 0.25:  # Lowered threshold for better recall
                return True
        
        # Method 2: Entity-based verification for structured data
        if key_entities:
            entity_support_count = 0
            for entity in key_entities:
                # Check for exact matches (case-insensitive)
                if entity.lower() in combined_sources or entity.upper() in combined_sources.upper():
                    entity_support_count += 1
                # Check for partial matches for complex entities
                elif any(part.lower() in combined_sources for part in entity.split() if len(part) > 2):
                    entity_support_count += 0.5
            
            if len(key_entities) > 0:
                entity_support_ratio = entity_support_count / len(key_entities)
                if entity_support_ratio >= 0.4:  # At least 40% of entities supported
                    return True
        
        # Method 3: Content-aware fuzzy matching
        sentence_words = sentence.lower().split()
        
        # Look for meaningful content overlaps
        content_words = [w for w in sentence_words if len(w) >= 4 and w.isalpha()]
        if content_words:
            content_matches = sum(1 for word in content_words if word in combined_sources.lower())
            content_ratio = content_matches / len(content_words)
            if content_ratio >= 0.3:  # At least 30% of content words match
                return True
        
        # Method 4: Exact phrase matching for high confidence
        # Split sentence into meaningful phrases (3+ words)
        words = sentence.split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3]).lower()
            if phrase in combined_sources.lower():
                return True
        
        # Method 5: Number and date matching for structured data
        numbers_in_sentence = re.findall(r'\b\d+\b', sentence)
        if numbers_in_sentence:
            number_matches = sum(1 for num in numbers_in_sentence if num in combined_sources)
            if len(numbers_in_sentence) > 0 and number_matches / len(numbers_in_sentence) >= 0.5:
                return True
        
        return False
    
    def _is_structured_data_sentence(self, sentence: str, search_results: List[Any]) -> bool:
        """Check if sentence appears to be about structured/tabular data."""
        
        
        # Common structured data indicators
        structured_indicators = [
            # Business/Finance
            'cost', 'price', 'revenue', 'profit', 'budget', 'expense', 'fee',
            'total', 'amount', 'value', 'balance', 'payment', 'invoice',
            
            # Data/Analytics
            'data', 'report', 'analysis', 'metric', 'statistic', 'count',
            'percentage', 'ratio', 'rate', 'average', 'maximum', 'minimum',
            
            # Scheduling/Planning
            'date', 'time', 'schedule', 'appointment', 'meeting', 'deadline',
            'calendar', 'event', 'period', 'duration', 'frequency',
            
            # IDs and References
            'id', 'code', 'number', 'reference', 'index', 'identifier',
            'serial', 'version', 'model', 'type', 'category', 'class',
            
            # Status and Classification
            'status', 'state', 'condition', 'level', 'grade', 'rank',
            'tier', 'priority', 'category', 'group', 'department'
        ]
        
        # Look for numeric patterns
        has_numbers = bool(re.search(r'\b\d+\b', sentence))
        has_dates = bool(re.search(r'\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}\b', sentence))
        has_codes = bool(re.search(r'\b[A-Z]{2,4}\d+\b', sentence.upper()))
        has_percentages = bool(re.search(r'\d+%', sentence))
        has_currency = bool(re.search(r'\$\d+', sentence))
        
        # Count structured indicators
        indicator_count = sum(1 for indicator in structured_indicators if indicator in sentence.lower())
        
        # Score the sentence
        structure_score = (
            indicator_count +
            (2 if has_numbers else 0) +
            (2 if has_dates else 0) +
            (3 if has_codes else 0) +
            (2 if has_percentages else 0) +
            (2 if has_currency else 0)
        )
        
        # More lenient threshold for structured data detection
        return structure_score >= 2
    
    def _generate_citations(self, search_results: List[Any]) -> List[Dict[str, Any]]:
        """Generate proper citations from search results."""
        
        citations = []
        
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 for citations
            chunk = result.chunk
            
            citation = {
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "source": chunk.metadata.get("document_name", "Unknown Document"),
                "page": chunk.metadata.get("page_number"),
                "confidence": round(result.combined_score, 3),
                "chunk_id": chunk.id
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_answer_confidence(self, search_results: List[Any]) -> float:
        """Calculate confidence score for the generated answer."""
        
        if not search_results:
            return 0.0
        
        # Base confidence on top retrieval scores
        top_scores = [result.combined_score for result in search_results[:3]]
        
        # Weight recent results more heavily
        weighted_score = (
            top_scores[0] * 0.5 + 
            (top_scores[1] if len(top_scores) > 1 else 0) * 0.3 +
            (top_scores[2] if len(top_scores) > 2 else 0) * 0.2
        )
        
        # Normalize to 0-1 range
        confidence = min(1.0, weighted_score)
        
        return round(confidence, 3)