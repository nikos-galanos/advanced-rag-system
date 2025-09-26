"""
Utility functions and helpers for the RAG application.
"""
import logging
import sys
from typing import Any, Dict, List
import re
import string


def setup_logger(name: str) -> logging.Logger:
    """Set up logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def clean_query(query: str) -> str:
    """Clean and normalize user query."""
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    # Remove special characters but keep basic punctuation
    query = re.sub(r'[^\w\s\?\!\.]', '', query)
    
    return query.strip()


def detect_intent(query: str) -> str:
    """Simple intent detection for queries."""
    query_lower = query.lower().strip()
    
    # Greeting patterns
    greeting_patterns = [
        r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
        r'\b(how are you|what\'s up|how\'s it going)\b'
    ]
    
    for pattern in greeting_patterns:
        if re.search(pattern, query_lower):
            return "greeting"
    
    # Question patterns
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
    if query_lower.startswith(tuple(question_words)) or query.endswith('?'):
        return "question"
    
    # Command patterns
    command_words = ['show', 'list', 'find', 'search', 'get', 'tell']
    if query_lower.startswith(tuple(command_words)):
        return "command"
    
    # Default to informational
    return "informational"


def should_trigger_search(query: str) -> bool:
    """Determine if query should trigger knowledge base search."""
    query_lower = query.lower().strip()
    
    # Don't search for greetings
    if detect_intent(query) == "greeting":
        return False
    
    # Don't search for very short queries
    if len(query.split()) < 2:
        return False
    
    # Don't search for common conversational patterns
    skip_patterns = [
        r'\b(thanks?|thank you|okay|ok|yes|no|maybe)\b$',
        r'\b(bye|goodbye|see you|farewell)\b'
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, query_lower):
            return False
    
    return True