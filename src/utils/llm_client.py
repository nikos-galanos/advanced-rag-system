"""
LLM client for various AI tasks including intent detection.
"""
import httpx
import json
from typing import Dict, Any, Optional
from src.config import settings
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class MistralClient:
    """Client for Mistral AI API."""
    
    def __init__(self):
        self.base_url = settings.mistral_base_url
        self.api_key = settings.mistral_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(self, messages: list, model: str = None, temperature: float = 0.1, max_tokens: int = 100) -> Optional[str]:
        """Generate chat completion using Mistral API."""
        if not model:
            model = settings.mistral_model
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling Mistral API: {str(e)}")
            return None
    
    async def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect query intent using LLM."""
        
        # Intent detection prompt
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that classifies user queries for a document search system.

Analyze the user query and determine:
1. Intent category (greeting, question, command, informational, conversational)
2. Whether it should trigger a document search (true/false)
3. Query type for response formatting (list, table, comparison, explanation, factual)
4. Confidence level (0.0-1.0)

Respond ONLY with a valid JSON object in this format:
{
    "intent": "question",
    "trigger_search": true,
    "query_type": "explanation",
    "confidence": 0.9,
    "reasoning": "User is asking for information that requires document search"
}

Intent categories:
- "greeting": Hello, hi, how are you, etc.
- "question": What, how, why, when, where questions
- "command": Show me, list, find, search, etc.
- "informational": Requests for specific information
- "conversational": Thank you, goodbye, yes/no responses

Query types:
- "list": Requests for lists or enumerations
- "table": Requests for comparisons or structured data
- "explanation": Requests for detailed explanations
- "factual": Simple fact-based questions
- "summary": Requests for summaries or overviews"""
            },
            {
                "role": "user",
                "content": f"Query: \"{query}\""
            }
        ]
        
        try:
            response = await self.chat_completion(messages, temperature=0.0, max_tokens=150)
            
            if response:
                # Parse JSON response
                try:
                    result = json.loads(response)
                    
                    # Validate required fields
                    required_fields = ["intent", "trigger_search", "query_type", "confidence"]
                    if all(field in result for field in required_fields):
                        logger.info(f"LLM intent detection: {result['intent']} (confidence: {result['confidence']})")
                        return result
                    else:
                        logger.warning(f"Invalid LLM response format: {response}")
                        return self._fallback_intent_detection(query)
                        
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response as JSON: {response}")
                    return self._fallback_intent_detection(query)
            
            # Fallback if API call failed
            return self._fallback_intent_detection(query)
            
        except Exception as e:
            logger.error(f"Error in LLM intent detection: {str(e)}")
            return self._fallback_intent_detection(query)
    
    def _fallback_intent_detection(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based intent detection."""
        from src.utils.helpers import detect_intent, should_trigger_search
        
        intent = detect_intent(query)
        trigger_search = should_trigger_search(query)
        
        # Determine query type
        query_lower = query.lower()
        if any(word in query_lower for word in ['list', 'show all', 'enumerate']):
            query_type = "list"
        elif any(word in query_lower for word in ['table', 'comparison', 'compare']):
            query_type = "table"
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            query_type = "summary"
        elif query.endswith('?'):
            query_type = "explanation" if len(query.split()) > 5 else "factual"
        else:
            query_type = "explanation"
        
        return {
            "intent": intent,
            "trigger_search": trigger_search,
            "query_type": query_type,
            "confidence": 0.6,  # Lower confidence for rule-based
            "reasoning": "Fallback rule-based detection",
            "method": "rule-based"
        }
    
    async def generate_embeddings(self, texts: list) -> Optional[list]:
        """Generate embeddings for text using Mistral API."""
        payload = {
            "model": settings.mistral_embed_model,
            "input": texts
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    return embeddings
                else:
                    logger.error(f"Mistral embeddings API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None