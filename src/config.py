"""
Configuration settings for the RAG system.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = "Advanced RAG System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Mistral AI API
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY")
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model: str = "mistral-large-latest"
    mistral_embed_model: str = "mistral-embed"
    
    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".pdf"]
    
    # Retrieval Settings
    top_k_retrieval: int = 10
    similarity_threshold: float = 0.7
    max_query_length: int = 500
    
    # Generation Settings
    max_tokens: int = 1000
    temperature: float = 0.1
    
    # Storage
    data_dir: str = "data"
    vectors_file: str = "data/vectors.pkl"
    metadata_file: str = "data/metadata.pkl"
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()