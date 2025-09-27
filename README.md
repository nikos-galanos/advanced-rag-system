# Advanced RAG System

An advanced Retrieval-Augmented Generation system built from scratch with hybrid semantic and keyword search capabilities, designed for enterprise document analysis and query processing.

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Design](#architecture-design)
- [PDF Processing & Chunking](#pdf-processing--chunking)
- [Query Processing Pipeline](#query-processing-pipeline)
- [Hybrid Retrieval Engine](#hybrid-retrieval-engine)
- [Generation & Safety](#generation--safety)
- [Installation & Setup](#installation--setup)
- [API Documentation](#api-documentation)
- [User Interface](#user-interface)
- [Technical Implementation](#technical-implementation)
- [Libraries & Dependencies](#libraries--dependencies)
- [Performance & Evaluation](#performance--evaluation)

## System Overview

This advanced RAG application implements a sophisticated document processing and query answering pipeline that combines semantic understanding with keyword-based retrieval. The system is designed to handle complex business documents, provide evidence-based answers with proper citations, and maintain high standards for accuracy and safety.

### Key Features

- **Hybrid Retrieval**: Combines Mistral embeddings (semantic) with custom BM25 implementation (keyword)
- **Table-Aware Processing**: Detects and processes structured data from PDF tables
- **LLM-Powered Query Understanding**: Intent detection and query transformation using Mistral AI
- **Evidence-Based Generation**: Hallucination filtering with source verification
- **Enterprise Safety**: PII detection, content filtering, and query refusal policies
- **No External Dependencies**: Custom implementations without third-party vector databases

## Architecture Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Document        │───▶│   Chunking &    │
│   Interface     │    │  Processing      │    │   Indexing      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                           │
                              ▼                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query           │───▶│   Hybrid        │
│   Interface     │    │  Processing      │    │   Retrieval     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                           │
                              ▼                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Generated     │◀───│  Answer          │◀───│   Context       │
│   Response      │    │  Generation      │    │   Building      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **Document Ingestion Engine** (`src/core/ingestion.py`)
   - Multi-method PDF extraction (PyPDF2 + pdfplumber)
   - Table detection and structured data processing
   - Semantic boundary chunking
   - Metadata enrichment and persistence

2. **Query Processing Pipeline** (`src/core/query.py`)
   - LLM-powered intent detection
   - Fallback mechanism for intent detection using keywords
   - Hybrid query transformation (rule-based + LLM enhancement)
   - Domain-aware synonym expansion
   - Safety filtering and query refusal policies

3. **Hybrid Retrieval Engine** (`src/core/retrieval.py`)
   - Semantic search using Mistral embeddings
   - Custom BM25 keyword search implementation
   - MMR (Maximal Marginal Relevance) re-ranking
   - Multi-query search variations

4. **Generation Engine** (`src/core/generation.py`)
   - Context-aware answer generation with template selection
   - Evidence-based filtering with hallucination detection
   - Citation system with confidence scoring
   - Template switching based on query intent

## PDF Processing & Chunking

### Design Considerations

The PDF processing system was designed to handle real-world business documents with complex layouts, tables, and mixed content types. Key considerations included:

#### **Multi-Method Extraction Strategy**
- **Primary Method (PyPDF2)**: Fast, reliable text extraction for standard documents
- **Fallback Method (pdfplumber)**: Advanced layout analysis for complex documents with tables
- **Rationale**: Business documents often contain structured data that requires specialized extraction

#### **Table Detection & Processing**
```python
# Table awareness implementation
def _analyze_tables(self, tables: List[List[List]]) -> List[Dict[str, Any]]:
    """
    Detects table structure and creates searchable metadata
    - Analyzes column types (numeric, date, text)
    - Generates searchable content summaries
    - Preserves table relationships for better retrieval
    """
```

**Considerations:**
- **Searchability**: Tables often contain critical business data (financial metrics, schedules, inventories)
- **Context Preservation**: Maintain relationships between table headers and data
- **Mixed Content**: Handle documents with both narrative text and structured data

#### **Chunking Algorithm**

```python
def _chunk_text_intelligently(self, text: str, filename: str, page_num: int, 
                             method: str, page_info: Dict[str, Any]) -> List[DocumentChunk]:
    """
    Semantic boundary-aware chunking with configurable overlap
    - Respects paragraph boundaries and sentence structure
    - Handles variable-length content appropriately
    - Maintains context while optimizing for retrieval
    """
```

**Key Considerations:**

1. **Semantic Boundaries**: 
   - Prioritize paragraph breaks over arbitrary character limits
   - Preserve sentence integrity to maintain meaning
   - Handle variable content types (narrative vs. structured)

2. **Overlap Strategy**:
   - 50-character overlap to maintain context continuity
   - Configurable chunk size (512 characters default) balancing context vs. precision
   - Special handling for very long paragraphs with sentence-level splitting

3. **Metadata Enrichment**:
   - Track extraction method for quality assessment
   - Preserve page numbers and document structure
   - Add content type flags (text, table, mixed)
   - Include word counts and structural indicators

4. **Quality Considerations**:
   - Filter out noise (page numbers, headers, footers)
   - Clean excessive whitespace while preserving structure
   - Handle special characters and encoding issues
   - Validate chunk quality before indexing

### Implementation Trade-offs

- **Performance vs. Accuracy**: Dual extraction methods add processing time but improve text quality
- **Chunk Size**: 512 characters balances context richness with retrieval precision
- **Storage**: Rich metadata increases storage requirements but improves search quality
- **Complexity**: Table awareness adds complexity but enables structured data queries

## Query Processing Pipeline

### Intent Detection System

# The system implements intelligent query understanding for optimal retrieval strategies:

```python
# LLM-powered intent detection with fallback
async def detect_intent(self, query: str) -> Dict[str, Any]:
    """
    Multi-stage intent detection:
    1. LLM analysis with structured JSON output
    2. Rule-based fallback for reliability
    3. Confidence scoring and validation
    """
```

**Query Types Detected:**
- **Factual**: Direct information requests requiring precise answers
- **List**: Enumeration requests for comprehensive retrieval
- **Comparison**: Multi-perspective analysis requiring diverse sources
- **Explanation**: In-depth analysis requiring contextual understanding

### Query Transformation Engine

```python
class HybridQueryTransformer:
    """
    Combines rule-based synonym expansion with LLM enhancement
    - Domain-aware synonym mapping (business, technical, academic)
    - LLM-powered query enhancement for complex requests
    - Multi-query variation generation for comprehensive search
    """
```

**Transformation Strategies:**
1. **Synonym Expansion**: Domain-specific term enrichment
2. **LLM Enhancement**: Context-aware query reformulation
3. **Search Variations**: Multiple query perspectives for better recall
4. **Keyword Extraction**: Core concept identification for hybrid search

## Hybrid Retrieval Engine

### Semantic Search Implementation

```python
class EmbeddingsEngine:
    """
    Embeddings management without external vector databases
    - Mistral API integration for high-quality embeddings
    - Persistent caching with pickle storage
    - Batch processing for efficiency
    - Similarity computation with cosine distance
    """
```

### Keyword Search (BM25)

```python
class BM25KeywordSearch:
    """
    Custom BM25 implementation built from scratch
    - Configurable parameters (k1=1.5, b=0.75)
    - Tokenization and preprocessing
    - Term frequency analysis with IDF calculation
    - Document length normalization
    """
```

**BM25 Configuration Rationale:**
- **k1=1.5**: Moderate term frequency saturation for business documents
- **b=0.75**: Standard document length normalization
- **Stopword Filtering**: Removes common words while preserving domain terms
- **Custom Tokenization**: Handles business terminology and abbreviations

### Search Fusion Strategy

```python
def _merge_and_rerank_results(self, semantic_results, keyword_results, 
                             semantic_weight=0.7, keyword_weight=0.3):
    """
    Intelligent score combination with MMR re-ranking
    - Weighted fusion of semantic and keyword scores
    - Diversity optimization to avoid redundant results
    - Context-aware re-ranking based on query type
    """
```

**Scoring Strategy:**
- **Default Weights**: 70% semantic, 30% keyword for balanced relevance
- **Query-Adaptive**: Factual queries increase keyword weight
- **MMR Re-ranking**: Ensures result diversity while maintaining relevance
- **Confidence Thresholding**: Filters low-quality matches

## Generation & Safety

### Template-Based Generation

The system uses query-type-specific templates for optimal response formatting:

```python
templates = {
    "factual": "Provide direct, factual answers based strictly on context",
    "list": "Create well-structured lists with clear explanations", 
    "comparison": "Present balanced comparisons with clear distinctions",
    "explanation": "Provide comprehensive explanations covering key aspects"
}
```

### Hallucination Prevention

```python
def _apply_hallucination_filter(self, answer: str, search_results: List[Any]):
    """
    Multi-layer fact verification:
    1. Entity extraction (dates, names, numbers, codes)
    2. Source text matching with fuzzy logic
    3. Structured data awareness for tables/lists
    4. Confidence scoring for each claim
    """
```

**Safety Mechanisms:**
- **Evidence Thresholding**: Minimum similarity requirements (0.5 default)
- **Source Verification**: Every claim must be traceable to source documents
- **PII Detection**: Automatic rejection of queries containing personal information
- **Content Filtering**: Medical/legal disclaimers for sensitive topics

### Citation System

```python
def _generate_citations(self, search_results: List[Any]) -> List[Dict[str, Any]]:
    """
    Comprehensive source attribution:
    - Document name and page number tracking
    - Confidence scoring for each citation
    - Content preview with character limits
    - Chunk-level traceability for debugging
    """
```

## Installation & Setup

### Prerequisites

- Python 3.9+
- Mistral AI API key
- Ideally 8GB+ RAM (for document processing)
- Ideally 1GB+ disk space (for embeddings cache)

### Quick Start

1. **Clone Repository**
```bash
git clone <repository-url>
cd advanced-rag-system
```

2. **Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
# Create .env file
echo "MISTRAL_API_KEY=your_api_key_here" > .env
```

4. **Start the System**
```bash
# Option 1: Auto-start (recommended)
streamlit run app.py

# Option 2: Manual start
# Terminal 1: Backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend  
streamlit run frontend/streamlit_ui.py
```

5. **Access the Interface**
- Open browser to: `http://localhost:8501`
- Upload PDF documents via the "Document Upload" tab
- Start querying via the "Chat with Documents" tab

### Configuration Options

Edit `src/config.py` to customize:

```python
# Document Processing
chunk_size: int = 512          # Characters per chunk
chunk_overlap: int = 50        # Overlap between chunks
max_file_size: int = 10MB      # Maximum PDF size

# Retrieval Settings  
top_k_retrieval: int = 10      # Results to retrieve
similarity_threshold: float = 0.7  # Minimum similarity

# Generation Settings
max_tokens: int = 1500         # Maximum response length
temperature: float = 0.1       # LLM creativity (0.0-1.0)
```

## API Documentation

### Core Endpoints

#### Document Ingestion
```http
POST /api/v1/ingest
Content-Type: multipart/form-data

Body: PDF files (multiple supported)
Response: Processing statistics and chunk counts
```

#### Query Processing
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are the main stakeholders?",
  "top_k": 5,
  "include_metadata": true
}
```

**Response Format:**
```json
{
  "query": "Original query text",
  "answer": "Generated response with evidence",
  "citations": [
    {
      "text": "Source text preview...", 
      "source": "document.pdf",
      "page": 1,
      "confidence": 0.85
    }
  ],
  "metadata": {
    "processing_time": 2.34,
    "chunks_retrieved": 5,
    "confidence": 0.92,
    "intent": "factual",
    "retrieval_results": {...},
    "generation_metadata": {...}
  }
}
```

#### System Statistics
```http
GET /api/v1/stats
Response: Dynamic system capabilities and metrics
```

## User Interface

### Design Philosophy

The Streamlit interface prioritizes **functionality over complexity**, designed specifically for technical demonstration

#### Main Features

1. **Collapsible Query History**
   - Hidden by default to maximize screen space
   - On-demand access to previous queries
   - Compact display with query previews

2. **Sub-tabbed Response Organization**
   - **Sources**: Expandable citations with confidence scores
   - **Metrics**: Performance data and retrieval statistics  
   - **Details**: Advanced processing information for debugging

3. **Progressive Disclosure**
   - Core answer always visible
   - Supporting information organized in logical hierarchy
   - Advanced features accessible


### Interface Components

- **Chat Interface**: Clean query input with immediate response display
- **Document Upload**: Drag-and-drop with progress tracking and validation
- **System Overview**: Dynamic capability reporting based on actual system state
- **Sidebar Controls**: Query parameters and system status monitoring

## Technical Implementation

### Code Organization

```
src/
├── api/                    # FastAPI routes and models
│   ├── routes.py          # Main API endpoints
│   └── models.py          # Pydantic request/response models
├── core/                  # Core RAG components
│   ├── ingestion.py       # PDF processing and chunking
│   ├── query.py           # Query processing and intent detection
│   ├── retrieval.py       # Hybrid search engine
│   ├── generation.py      # Answer generation and safety
│   ├── embeddings.py      # Semantic search implementation
│   └── keyword_search.py  # BM25 implementation
├── utils/                 # Utilities and helpers
│   ├── llm_client.py      # Mistral AI integration
│   ├── query_transformer.py  # Query enhancement
│   └── helpers.py         # Common utilities
├── config.py              # Configuration management
└── frontend/              # Streamlit UI
    └── streamlit_ui.py    # Complete user interface
```

### Error Handling Strategy

```python
# Graceful degradation with informative feedback
try:
    primary_extraction = pdfplumber_extract(pdf_content)
except Exception as e:
    logger.warning(f"pdfplumber failed: {e}")
    fallback_extraction = pypdf2_extract(pdf_content)
```

## Libraries & Dependencies

### Core Dependencies

#### Backend Framework
- **FastAPI (0.104.1)**: Modern, high-performance API framework

#### Document Processing
- **PyPDF2 (3.0.1)**: Primary PDF text extraction
  - Fast, reliable extraction for standard documents

- **pdfplumber (0.10.3)**: Advanced PDF analysis
  - Table detection and extraction capabilities
  - Layout-aware text processing

#### Machine Learning & Embeddings
- **sentence-transformers (2.2.2)**: Fallback embedding generation
- **numpy (1.24.3)**: Numerical computing for vector operations
- **scikit-learn (1.3.0)**: Similarity calculations and clustering

#### LLM Integration
- **httpx (0.25.0)**: Async HTTP client for Mistral AI API
- **requests (2.31.0)**: Synchronous HTTP operations

#### User Interface
- **streamlit (1.28.1)**: Interactive web application framework

#### Utilities
- **python-dotenv (1.0.0)**: Environment variable management
- **loguru (0.7.2)**: Advanced logging with structured output
- **pydantic (2.5.0)**: Data validation and settings management

## Performance & Evaluation

### Retrieval Quality Metrics

The system implements comprehensive evaluation of retrieval performance:

#### **Relevance Scoring**
- **Semantic Similarity**: Cosine similarity with Mistral embeddings
- **Keyword Matching**: BM25 scoring with document frequency analysis
- **Hybrid Scoring**: Weighted combination optimized for query types
- **Confidence Thresholding**: Minimum similarity requirements (0.5 default)

#### **Answer Quality Assessment**
- **Evidence Verification**: Source attribution for every claim
- **Hallucination Detection**: Multi-layer fact checking against sources
- **Citation Accuracy**: Precise source tracking with page numbers
- **Response Completeness**: Coverage of query requirements


## License

This project is developed as a technical demonstration portfolio project. All rights reserved.
