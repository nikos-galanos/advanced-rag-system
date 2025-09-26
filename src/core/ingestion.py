"""
Document ingestion and processing for the RAG system.
"""
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import io

import PyPDF2
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from fastapi import UploadFile
import numpy as np

from src.config import settings
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class DocumentChunk:
    """Represents a chunk of text from a document."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.id = self._generate_id(text, metadata)
        self.text = text
        self.metadata = metadata
        self.embedding: Optional[np.ndarray] = None
        self.created_at = datetime.now()
    
    def _generate_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for chunk."""
        content = f"{text[:100]}{metadata.get('document_name', '')}{metadata.get('page_number', 0)}{metadata.get('chunk_index', 0)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class DocumentProcessor:
    """Handles document ingestion, chunking, and metadata extraction."""
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data if available
        self._load_existing_data()
        
        logger.info(f"DocumentProcessor initialized with {len(self.chunks)} existing chunks")
    
    async def process_documents(self, files: List[UploadFile]) -> int:
        """Process multiple PDF files and return total chunks created."""
        logger.info(f"Processing {len(files)} documents")
        total_chunks = 0
        new_chunks_count = 0
        
        for file in files:
            try:
                # Check if document already processed
                file_content = await file.read()
                file_hash = hashlib.md5(file_content).hexdigest()
                
                if file.filename in self.documents and self.documents[file.filename].get("file_hash") == file_hash:
                    logger.info(f"Document {file.filename} already processed, skipping")
                    continue
                
                chunks_created = await self._process_single_document(file, file_content)
                total_chunks += chunks_created
                new_chunks_count += chunks_created
                logger.info(f"Processed {file.filename}: {chunks_created} new chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                # Don't raise, continue with other files
                continue
        
        # Save processed data if we have new chunks
        if new_chunks_count > 0:
            self._save_data()
            logger.info(f"Saved {new_chunks_count} new chunks. Total: {len(self.chunks)}")
        
        return new_chunks_count
    
    async def _process_single_document(self, file: UploadFile, content: bytes) -> int:
        """Process a single PDF document."""
        # Extract text from PDF
        text_pages = self._extract_text_from_pdf(content, file.filename)
        
        if not text_pages:
            logger.warning(f"No text extracted from {file.filename}")
            return 0
        
        # Create chunks
        chunks = self._create_chunks(text_pages, file.filename)
        
        # Store document metadata
        self.documents[file.filename] = {
            "filename": file.filename,
            "size": len(content),
            "pages": len(text_pages),
            "chunks": len(chunks),
            "processed_at": datetime.now(),
            "file_hash": hashlib.md5(content).hexdigest()
        }
        
        # Add chunks to collection
        # Remove old chunks from same document
        self.chunks = [c for c in self.chunks if c.metadata.get("document_name") != file.filename]
        self.chunks.extend(chunks)
        
        return len(chunks)
    
    def _extract_text_from_pdf(self, content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using PyPDF2 with fallback to pdfplumber."""
        pages = []
        
        try:
            # Method 1: PyPDF2
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():  # Only add non-empty pages
                        pages.append({
                            "page_number": page_num + 1,
                            "text": text,
                            "extraction_method": "PyPDF2"
                        })
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1} with PyPDF2: {str(e)}")
                    continue
            
            logger.info(f"Extracted {len(pages)} pages from {filename} using PyPDF2")
            
            # If we got good results, return them
            if pages:
                return pages
                
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {filename}: {str(e)}")
        
        # Method 2: Acts as a FALLBACK: pdfplumber (more robust for complex layouts)
        if pdfplumber:
            try:
                pdf_file = io.BytesIO(content)
                with pdfplumber.open(pdf_file) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text and text.strip():
                                pages.append({
                                    "page_number": page_num + 1,
                                    "text": text,
                                    "extraction_method": "pdfplumber"
                                })
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num + 1} with pdfplumber: {str(e)}")
                            continue
                
                logger.info(f"Extracted {len(pages)} pages from {filename} using pdfplumber")
                
            except Exception as e2:
                logger.warning(f"pdfplumber also failed for {filename}: {str(e2)}")
        
        if not pages:
            raise Exception(f"Could not extract any text from {filename}")
        
        return pages
    
    def _create_chunks(self, pages: List[Dict[str, Any]], filename: str) -> List[DocumentChunk]:
        """Create text chunks from extracted pages with smart boundary detection."""
        chunks = []
        
        for page in pages:
            text = page["text"]
            page_chunks = self._chunk_text_intelligently(
                text, 
                filename, 
                page["page_number"],
                page.get("extraction_method", "unknown")
            )
            chunks.extend(page_chunks)
        
        logger.info(f"Created {len(chunks)} chunks for {filename}")
        return chunks
    
    def _chunk_text_intelligently(self, text: str, filename: str, page_num: int, method: str) -> List[DocumentChunk]:
        """Create chunks with semantic boundary detection."""
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        if not text.strip():
            return chunks
        
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If no paragraph breaks, split by single newlines
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > settings.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(), filename, page_num, chunk_index, method
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle very long paragraphs
                if len(paragraph) > settings.chunk_size:
                    # Split long paragraph by sentences
                    sentences = self._split_by_sentences(paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 > settings.chunk_size:
                            if temp_chunk.strip():
                                chunk = self._create_chunk(
                                    temp_chunk.strip(), filename, page_num, chunk_index, method
                                )
                                chunks.append(chunk)
                                chunk_index += 1
                            temp_chunk = sentence
                        else:
                            temp_chunk += " " + sentence if temp_chunk else sentence
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(current_chunk.strip(), filename, page_num, chunk_index, method)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, filename: str, page_num: int, chunk_index: int, method: str) -> DocumentChunk:
        """Create a DocumentChunk with comprehensive metadata."""
        metadata = {
            "document_name": filename,
            "page_number": page_num,
            "chunk_index": chunk_index,
            "extraction_method": method,
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_questions": "?" in text,
            "has_numbers": any(char.isdigit() for char in text)
        }
        
        return DocumentChunk(text, metadata)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
            
        # Remove excessive whitespace BUT preserve structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be headers/footers/page numbers
            if len(line) > 2 and not (line.isdigit() and len(line) < 4):
                cleaned_lines.append(line)
        
        # Join with single newlines and clean up spacing
        cleaned = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace but keep paragraph structure
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 15:
                # Look ahead to avoid splitting on abbreviations
                remaining = text[len(current):].lstrip()
                if remaining and (remaining[0].isupper() or remaining[0] in '\n"\''):
                    sentences.append(current.strip())
                    current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if s.strip()]
    
    def _save_data(self):
        """Save chunks and documents to disk."""
        try:
            # Ensure data folder exists
            os.makedirs(os.path.dirname(settings.metadata_file), exist_ok=True)
            
            # Prepare chunks data for serialization
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "created_at": chunk.created_at,
                    "embedding": chunk.embedding.tolist() if chunk.embedding is not None else None
                }
                chunks_data.append(chunk_dict)
            
            # Save to file
            data = {
                "chunks": chunks_data,
                "documents": self.documents,
                "last_updated": datetime.now()
            }
            
            with open(settings.metadata_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.chunks)} chunks and {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def _load_existing_data(self):
        """Load existing chunks and documents."""
        try:
            if os.path.exists(settings.metadata_file):
                with open(settings.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Reconstruct chunks
                for chunk_data in data.get("chunks", []):
                    chunk = DocumentChunk(chunk_data["text"], chunk_data["metadata"])
                    chunk.id = chunk_data["id"]
                    chunk.created_at = chunk_data["created_at"]
                    if chunk_data.get("embedding"):
                        chunk.embedding = np.array(chunk_data["embedding"])
                    self.chunks.append(chunk)
                
                self.documents = data.get("documents", {})
                logger.info(f"Loaded {len(self.chunks)} existing chunks from {len(self.documents)} documents")
                
        except Exception as e:
            logger.warning(f"Could not load existing data: {str(e)}")
            # Initialize empty collections
            self.chunks = []
            self.documents = {}
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)
    
    def get_last_update(self) -> Optional[datetime]:
        """Get last update timestamp."""
        if not self.documents:
            return None
        return max(doc["processed_at"] for doc in self.documents.values())
    
    def get_chunks(self) -> List[DocumentChunk]:
        """Get all chunks."""
        return self.chunks
    
    def search_chunks_by_metadata(self, **kwargs) -> List[DocumentChunk]:
        """Search chunks by metadata criteria."""
        matching_chunks = []
        
        for chunk in self.chunks:
            matches = True
            for key, value in kwargs.items():
                if chunk.metadata.get(key) != value:
                    matches = False
                    break
            if matches:
                matching_chunks.append(chunk)
        
        return matching_chunks