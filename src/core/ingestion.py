"""
Document ingestion and processing for the RAG system with table awareness.
"""
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import io
import re

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
    """Handles document ingestion, chunking, and metadata extraction with table awareness."""
    
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
                continue
        
        # Save processed data if we have new chunks
        if new_chunks_count > 0:
            self._save_data()
            logger.info(f"Saved {new_chunks_count} new chunks. Total: {len(self.chunks)}")
        
        return new_chunks_count
    
    async def _process_single_document(self, file: UploadFile, content: bytes) -> int:
        """Process a single PDF document."""
        # Extract text from PDF with table detection
        text_pages = self._extract_text_from_pdf(content, file.filename)
        
        if not text_pages:
            logger.warning(f"No text extracted from {file.filename}")
            return 0
        
        # Create chunks with table awareness
        chunks = self._create_chunks(text_pages, file.filename)
        
        # Store document metadata
        self.documents[file.filename] = {
            "filename": file.filename,
            "size": len(content),
            "pages": len(text_pages),
            "chunks": len(chunks),
            "processed_at": datetime.now(),
            "file_hash": hashlib.md5(content).hexdigest(),
            "tables_detected": sum(page.get("tables_detected", 0) for page in text_pages),
            "pages_with_tables": sum(1 for page in text_pages if page.get("has_structured_data", False))
        }
        
        # Add chunks to collection (remove old ones if reprocessing)
        self.chunks = [c for c in self.chunks if c.metadata.get("document_name") != file.filename]
        self.chunks.extend(chunks)
        
        return len(chunks)
    
    def _extract_text_from_pdf(self, content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using multiple methods with table detection."""
        pages = []
        
        try:
            # Method 1: PyPDF2 (faster, basic extraction)
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": page_num + 1,
                            "text": text,
                            "extraction_method": "PyPDF2",
                            "tables_detected": 0,  # PyPDF2 can't detect tables
                            "has_structured_data": False
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
        
        # Method 2: pdfplumber (more robust for complex layouts + table detection)
        if pdfplumber:
            try:
                pdf_file = io.BytesIO(content)
                with pdfplumber.open(pdf_file) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            
                            # Extract and analyze tables
                            tables = page.extract_tables()
                            table_count = len(tables) if tables else 0
                            
                            # Enhance text with table context
                            enhanced_text = text
                            if table_count > 0:
                                enhanced_text += f"\n\n[Document contains {table_count} table(s) with structured data on this page]"
                                
                                # Add table summaries to make them searchable
                                for i, table in enumerate(tables):
                                    if table and len(table) > 0:
                                        headers = table[0] if table[0] else []
                                        row_count = len(table) - 1 if len(table) > 1 else 0
                                        
                                        if headers:
                                            clean_headers = [str(h) for h in headers if h]
                                            enhanced_text += f"\n[Table {i+1} headers: {', '.join(clean_headers)}; {row_count} data rows]"
                            
                            if enhanced_text and enhanced_text.strip():
                                pages.append({
                                    "page_number": page_num + 1,
                                    "text": enhanced_text,
                                    "extraction_method": "pdfplumber",
                                    "tables_detected": table_count,
                                    "has_structured_data": table_count > 0,
                                    "table_info": self._analyze_tables(tables) if tables else []
                                })
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num + 1} with pdfplumber: {str(e)}")
                            continue
                
                logger.info(f"Extracted {len(pages)} pages from {filename} using pdfplumber with table detection")
                
            except Exception as e2:
                logger.warning(f"pdfplumber also failed for {filename}: {str(e2)}")
        
        if not pages:
            raise Exception(f"Could not extract any text from {filename}")
        
        return pages
    
    def _analyze_tables(self, tables: List[List[List]]) -> List[Dict[str, Any]]:
        """Analyze extracted tables to create searchable metadata."""
        table_info = []
        
        for i, table in enumerate(tables):
            if not table or len(table) == 0:
                continue
                
            try:
                headers = table[0] if table[0] else []
                data_rows = table[1:] if len(table) > 1 else []
                
                # Clean headers
                clean_headers = [str(h).strip() for h in headers if h and str(h).strip()]
                
                # Analyze table content
                info = {
                    "table_index": i,
                    "headers": clean_headers,
                    "row_count": len(data_rows),
                    "column_count": len(clean_headers),
                    "contains_numbers": self._table_contains_numbers(data_rows),
                    "likely_data_types": self._detect_column_types(data_rows, len(clean_headers)),
                    "searchable_content": self._create_table_search_text(clean_headers, data_rows)
                }
                
                table_info.append(info)
                
            except Exception as e:
                logger.warning(f"Error analyzing table {i}: {str(e)}")
                continue
        
        return table_info
    
    def _table_contains_numbers(self, data_rows: List[List]) -> bool:
        """Check if table contains numerical data."""
        for row in data_rows[:5]:  # Check first 5 rows
            for cell in row:
                if cell and str(cell).strip():
                    # Try to find numbers in cell
                    cell_str = str(cell).replace(',', '').replace('$', '').replace('%', '')
                    try:
                        float(cell_str)
                        return True
                    except ValueError:
                        # Check for number patterns
                        if re.search(r'\d+', str(cell)):
                            return True
        return False
    
    def _detect_column_types(self, data_rows: List[List], column_count: int) -> List[str]:
        """Detect likely data types for each column."""
        if not data_rows or column_count == 0:
            return []
        
        types = []
        for col_idx in range(min(column_count, 10)):  # max 10 cols
            column_values = []
            for row in data_rows[:10]:  # Sample 10 rows
                if col_idx < len(row) and row[col_idx]:
                    column_values.append(str(row[col_idx]).strip())
            
            if not column_values:
                types.append("empty")
                continue
            
            # Analyze column content
            numeric_count = 0
            date_count = 0
            text_count = 0
            
            for value in column_values:
                if not value:
                    continue
                    
                # Check for numbers
                clean_value = value.replace(',', '').replace('$', '').replace('%', '')
                try:
                    float(clean_value)
                    numeric_count += 1
                    continue
                except ValueError:
                    pass
                
                # Check for dates
                if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value) or \
                   re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', value):
                    date_count += 1
                    continue
                
                text_count += 1
            
            # Determine predominant type by prevalence
            total = len(column_values)
            if numeric_count > total * 0.6:
                types.append("numeric")
            elif date_count > total * 0.6:
                types.append("date")
            else:
                types.append("text")
        
        return types
    
    def _create_table_search_text(self, headers: List[str], data_rows: List[List]) -> str:
        """Create searchable text representation of table."""
        search_parts = []
        
        # Add headers as searchable content
        if headers:
            search_parts.append(f"Table columns: {', '.join(headers)}")
        
        # Add sample data for search context (few rows)
        sample_rows = data_rows[:3] if data_rows else []
        for i, row in enumerate(sample_rows):
            if row and any(cell for cell in row if cell):
                row_text = " | ".join(str(cell) for cell in row if cell)
                search_parts.append(f"Row {i+1}: {row_text}")
        
        return "; ".join(search_parts)
    
    def _create_chunks(self, pages: List[Dict[str, Any]], filename: str) -> List[DocumentChunk]:
        """Create text chunks from extracted pages with tables."""
        chunks = []
        
        for page in pages:
            text = page["text"]
            page_chunks = self._chunk_text_intelligently(
                text, 
                filename, 
                page["page_number"],
                page.get("extraction_method", "unknown"),
                page  # Pass full page info for table awareness
            )
            chunks.extend(page_chunks)
        
        return chunks
    
    def _chunk_text_intelligently(self, text: str, filename: str, page_num: int, method: str, page_info: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Create chunks with semantic boundary detection and table awareness."""
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        if not text.strip():
            return chunks
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If no paragraph breaks, split by single newlines
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > settings.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(), filename, page_num, chunk_index, method, page_info
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
                                    temp_chunk.strip(), filename, page_num, chunk_index, method, page_info
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
            chunk = self._create_chunk(current_chunk.strip(), filename, page_num, chunk_index, method, page_info)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, filename: str, page_num: int, chunk_index: int, method: str, page_info: Dict[str, Any] = None) -> DocumentChunk:
        """Create a DocumentChunk with comprehensive metadata including table awareness."""
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
        
        # Add table-aware metadata if page info is available
        if page_info:
            metadata.update({
                "tables_on_page": page_info.get("tables_detected", 0),
                "contains_structured_data": page_info.get("has_structured_data", False),
                "table_info": page_info.get("table_info", [])
            })
            
            # Enhanced text analysis for table-containing chunks
            if page_info.get("has_structured_data", False):
                metadata["content_type"] = "mixed_text_and_tables"
                metadata["data_heavy"] = page_info.get("tables_detected", 0) > 1
                
                # Add searchable table content to metadata
                table_content = []
                for table_meta in page_info.get("table_info", []):
                    table_content.append(table_meta.get("searchable_content", ""))
                metadata["table_search_content"] = " | ".join(table_content)
            else:
                metadata["content_type"] = "text_only"
        
        return DocumentChunk(text, metadata)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
            
        # Remove excessive whitespace while preserving structure
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
    
    def get_table_statistics(self) -> Dict[str, Any]:
        """Get statistics about table detection across all documents."""
        total_tables = 0
        pages_with_tables = 0
        chunks_with_tables = 0
        
        for doc_info in self.documents.values():
            total_tables += doc_info.get("tables_detected", 0)
            pages_with_tables += doc_info.get("pages_with_tables", 0)
        
        for chunk in self.chunks:
            if chunk.metadata.get("contains_structured_data", False):
                chunks_with_tables += 1
        
        return {
            "total_tables_detected": total_tables,
            "pages_with_tables": pages_with_tables,
            "chunks_with_table_content": chunks_with_tables,
            "documents_with_tables": sum(1 for doc in self.documents.values() if doc.get("tables_detected", 0) > 0)
        }