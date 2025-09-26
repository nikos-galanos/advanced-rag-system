"""
Advanced RAG System Streamlit UI - Fixed Version
"""
import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Configure page
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .citation-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metadata-box {
        background-color: #e2e3e5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .answer-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Semantic + Keyword Search with LLM-Powered Query Processing</p>', unsafe_allow_html=True)
    
    # Sidebar for system info and settings
    with st.sidebar:
        st.header("📊 System Status")
        
        # Get system stats
        stats = get_system_stats()
        if stats:
            display_system_stats(stats)
        
        st.markdown("---")
        
        st.header("⚙️ Query Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        include_metadata = st.checkbox("Include detailed metadata", value=True)
        show_advanced = st.checkbox("Show advanced info", value=False)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat with Documents", "📁 Document Upload", "🔧 System Overview"])
    
    with tab1:
        chat_interface(top_k, include_metadata, show_advanced)
    
    with tab2:
        document_upload_interface()
    
    with tab3:
        system_overview_interface()

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None

def display_system_stats(stats: Dict[str, Any]):
    """Display system statistics in sidebar."""
    docs = stats.get("documents", {})
    retrieval = stats.get("retrieval", {})
    tables = stats.get("table_awareness", {})
    
    st.metric("Documents", docs.get("total_documents", 0))
    st.metric("Text Chunks", docs.get("total_chunks", 0))
    st.metric("Embeddings Cached", retrieval.get("embeddings_cached", 0))
    
    # Index status
    status = "🟢 Ready" if retrieval.get("index_built", False) else "🔴 Not Built"
    st.write(f"**Index Status:** {status}")
    
    # Table awareness
    if tables.get("total_tables_detected", 0) > 0:
        st.write(f"**Tables Detected:** {tables['total_tables_detected']}")

def chat_interface(top_k: int, include_metadata: bool, show_advanced: bool):
    """Main chat interface with collapsible sidebar history."""
    # Initialize chat history and current query state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    if "show_history_sidebar" not in st.session_state:
        st.session_state.show_history_sidebar = False
    
    # History toggle button
    if st.session_state.chat_history:
        col_toggle1, col_toggle2, col_toggle3 = st.columns([1, 2, 1])
        with col_toggle2:
            toggle_text = "📋 Hide History" if st.session_state.show_history_sidebar else f"📋 Show History ({len(st.session_state.chat_history)} queries)"
            if st.button(toggle_text, use_container_width=True):
                st.session_state.show_history_sidebar = not st.session_state.show_history_sidebar
                st.rerun()
    
    # Create layout based on history visibility
    if st.session_state.show_history_sidebar and st.session_state.chat_history:
        col1, col2 = st.columns([1, 4])  # Smaller history column when shown
        
        with col1:
            st.markdown("#### 📋 History")
            
            # Compact history display
            for i, (query, response) in enumerate(reversed(st.session_state.chat_history)):
                # Even more compact display
                display_query = query[:30] + "..." if len(query) > 30 else query
                query_num = len(st.session_state.chat_history) - i
                
                if st.button(f"Q{query_num}", key=f"history_{i}", help=query, use_container_width=True):
                    st.session_state.current_query = query
                    st.session_state.current_response = response
                    st.rerun()
                
                # Show truncated query below button
                st.caption(display_query)
                st.markdown("---")
            
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_query = None
                st.session_state.current_response = None
                st.session_state.show_history_sidebar = False
                st.rerun()
        
        main_col = col2
    else:
        # Full width when history is hidden
        main_col = st.container()
    
    with main_col:
        st.header("💬 Ask Questions About Your Documents")
        
        # Main query input at the top
        with st.form(key="main_query_form", clear_on_submit=True):
            query_input = st.text_area(
                "Ask a question about your documents:",
                placeholder="Enter your question here...",
                height=100,
                key="main_query_input"
            )
            
            col_a, col_b, col_c = st.columns([2, 1, 2])
            with col_b:
                submit_query = st.form_submit_button("🔍 Ask", type="primary", use_container_width=True)
        
        # Process new query
        if submit_query and query_input.strip():
            with st.spinner("Processing your query using advanced RAG pipeline..."):
                response = query_documents(query_input.strip(), top_k, include_metadata)
                
                if response:
                    # Add to history if this is a new query (not from history)
                    if st.session_state.current_query != query_input.strip():
                        st.session_state.chat_history.append((query_input.strip(), response))
                    
                    # Set as current
                    st.session_state.current_query = query_input.strip()
                    st.session_state.current_response = response
                else:
                    error_response = {"error": "Failed to get response from the system."}
                    st.session_state.current_query = query_input.strip()
                    st.session_state.current_response = error_response
                
                st.rerun()
        
        # Display current query and response
        if st.session_state.current_query and st.session_state.current_response:
            st.markdown("---")
            
            # Show current query
            st.markdown(f"**Current Query:** {st.session_state.current_query}")
            
            # Show response
            display_assistant_response(st.session_state.current_response, show_advanced)
        
        elif not st.session_state.current_query:
            # Welcome message when no query yet
            st.markdown("---")
            st.info("👆 Enter your question above to get started with the advanced RAG system!")
            
            # Show capabilities
            st.markdown("""
            **What you can ask about:**
            - Specific information from your uploaded documents
            - Complex analysis requiring multiple sources
            - Comparisons and relationships in the data
            - Detailed explanations of concepts from the documents
            """)
            
            # Show system status
            if st.session_state.chat_history:
                st.success(f"📊 **System Status:** Ready | {len(st.session_state.chat_history)} queries completed")

def display_assistant_response(response: Dict[str, Any], show_advanced: bool):
    """Display assistant response with organized sub-tabs."""
    if "error" in response:
        st.markdown(f'<div class="error-box">❌ {response["error"]}</div>', unsafe_allow_html=True)
        return
    
    # Main answer - prominently displayed
    answer = response.get("answer", "No answer provided.")
    st.markdown(f"""
    <div class="answer-section">
        <h4>📝 Answer</h4>
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Create sub-tabs for additional information
    tab1, tab2, tab3 = st.tabs(["📚 Sources", "📊 Metrics", "🔍 Details"])
    
    with tab1:
        # Citations with expandable sections
        citations = response.get("citations", [])
        if citations:
            st.markdown("**Sources and Evidence:**")
            for i, citation in enumerate(citations, 1):
                source = citation.get("source", "Unknown")
                confidence = citation.get("confidence", 0)
                text_preview = citation.get("text", "")[:200] + "..." if len(citation.get("text", "")) > 200 else citation.get("text", "")
                
                with st.expander(f"Source {i}: {source} (Confidence: {confidence:.3f})", expanded=False):
                    st.markdown(f"**Document:** {source}")
                    st.markdown(f"**Relevance Score:** {confidence:.3f}")
                    if citation.get("page"):
                        st.markdown(f"**Page:** {citation['page']}")
                    st.markdown(f"**Content Preview:**")
                    st.markdown(f'*"{text_preview}"*')
                    if citation.get("chunk_id"):
                        st.caption(f"Chunk ID: {citation['chunk_id']}")
        else:
            st.info("No sources available for this response.")
    
    with tab2:
        # Processing metrics and performance
        metadata = response.get("metadata", {})
        processing_time = response.get("processing_time", 0)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col2:
            chunks = metadata.get("retrieval_results", {}).get("chunks_retrieved", 0)
            st.metric("Chunks Retrieved", chunks)
        with col3:
            confidence = metadata.get("generation_metadata", {}).get("confidence", 0)
            st.metric("Answer Confidence", f"{confidence:.3f}")
        with col4:
            intent = metadata.get("intent", "Unknown")
            st.metric("Query Type", intent.title())
        
        # Retrieval performance
        if "retrieval_results" in metadata:
            st.markdown("**🎯 Retrieval Performance:**")
            retr = metadata["retrieval_results"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• **Semantic Score:** {retr.get('top_semantic_score', 0):.3f}")
                st.write(f"• **Combined Score:** {retr.get('top_combined_score', 0):.3f}")
                chunks_with_tables = retr.get('chunks_with_tables', 0)
                st.write(f"• **Chunks with Tables:** {chunks_with_tables}")
            with col2:
                st.write(f"• **Keyword Score:** {retr.get('top_keyword_score', 0):.3f}")
                st.write(f"• **Average Similarity:** {retr.get('avg_similarity', 0):.3f}")
        
        # Generation quality
        if "generation_metadata" in metadata:
            st.markdown("**⚡ Generation Quality:**")
            gen = metadata["generation_metadata"]
            halluc = gen.get("hallucination_check", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• **Template Used:** {gen.get('template_used', 'N/A')}")
                st.write(f"• **Evidence Quality:** {gen.get('evidence_quality', 'N/A')}")
                st.write(f"• **Citations Generated:** {gen.get('citation_count', 0)}")
            with col2:
                total_sentences = halluc.get('total_sentences', 0)
                supported = halluc.get('supported_sentences', 0)
                st.write(f"• **Supported Sentences:** {supported}/{total_sentences}")
                if halluc.get("filter_applied", False):
                    st.write("• ⚠️ **Hallucination filter applied**")
                else:
                    st.write("• ✅ **No filtering needed**")
                
                if halluc.get("structured_data_detected", False):
                    st.write("• 📊 **Structured data detected**")
    
    with tab3:
        # Advanced processing details (only if requested)
        if show_advanced and metadata:
            st.markdown("**🔧 Query Processing Pipeline:**")
            
            # Query transformation
            if "transformation_results" in metadata:
                trans = metadata["transformation_results"]
                with st.expander("🔄 Query Enhancement", expanded=False):
                    st.markdown("**Original vs Enhanced Query:**")
                    st.code(f"""
Original: {trans.get('original_query', 'N/A')}
Enhanced: {trans.get('transformed_query', 'N/A')}
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Intent:** {metadata.get('intent', 'N/A')}")
                        st.write(f"**Confidence:** {metadata.get('confidence', 0):.1%}")
                        st.write(f"**Method:** {metadata.get('detection_method', 'N/A')}")
                    with col2:
                        strategy = trans.get('transformation_strategy', {})
                        st.write(f"**Approach:** {strategy.get('approach', 'N/A')}")
                        st.write(f"**Domain Aware:** {strategy.get('domain_aware', 'N/A')}")
                    
                    variations = trans.get('search_variations', [])
                    if variations:
                        st.write("**Search Variations Generated:**")
                        for i, var in enumerate(variations[:3], 1):
                            st.write(f"{i}. `{var}`")
            
            # Search strategy
            if "retrieval_results" in metadata:
                retr = metadata["retrieval_results"]
                strategy = retr.get("search_strategy", {})
                
                with st.expander("🎯 Search Strategy", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Semantic Weight:** {strategy.get('semantic_weight', 0):.1%}")
                        st.write(f"**Keyword Weight:** {strategy.get('keyword_weight', 0):.1%}")
                        st.write(f"**Min Similarity:** {strategy.get('min_similarity', 0):.3f}")
                    with col2:
                        st.write(f"**Rerank Method:** {strategy.get('rerank_method', 'N/A')}")
                        st.write(f"**Diversity Threshold:** {strategy.get('diversity_threshold', 0):.3f}")
                        st.write(f"**Multi-pass Search:** {strategy.get('multi_pass_search', False)}")
            
            # System information
            with st.expander("🖥️ System Information", expanded=False):
                st.write(f"**Available Chunks:** {metadata.get('available_chunks', 0)}")
                table_stats = metadata.get('table_statistics', {})
                if table_stats:
                    st.write("**Table Detection:**")
                    st.write(f"• Total tables: {table_stats.get('total_tables_detected', 0)}")
                    st.write(f"• Documents with tables: {table_stats.get('documents_with_tables', 0)}")
                    st.write(f"• Chunks with table content: {table_stats.get('chunks_with_table_content', 0)}")
            
            # Full metadata dump
            with st.expander("📋 Complete Metadata", expanded=False):
                st.json(metadata)
        else:
            st.info("💡 Enable 'Show advanced info' in the sidebar for detailed processing information.")

def query_documents(query: str, top_k: int, include_metadata: bool) -> Dict[str, Any]:
    """Query the RAG system."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "include_metadata": include_metadata
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def document_upload_interface():
    """Document upload interface with persistent feedback."""
    st.header("📁 Upload PDF Documents")
    
    # Initialize upload state
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None
    if "upload_result" not in st.session_state:
        st.session_state.upload_result = None
    
    st.markdown("""
    <div class="feature-box">
        <h4>🚀 Advanced Document Processing Features:</h4>
        <ul>
            <li><strong>Multi-method extraction:</strong> PyPDF2 + pdfplumber for robust text extraction</li>
            <li><strong>Table awareness:</strong> Detects and processes structured data from tables</li>
            <li><strong>Intelligent chunking:</strong> Semantic boundary detection for optimal retrieval</li>
            <li><strong>Automatic indexing:</strong> Builds hybrid semantic + keyword search indices</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display persistent status messages
    if st.session_state.upload_status == "success" and st.session_state.upload_result:
        result = st.session_state.upload_result
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="success-box">
                ✅ <strong>Upload Successful!</strong><br>
                • Processed {result.get('files_processed', 0)} files<br>
                • Created {result.get('chunks_created', 0)} searchable chunks<br>
                • Processing time: {result.get('processing_time', 0):.2f} seconds<br>
                • Search indices built and ready for queries
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✖️ Dismiss", key="dismiss_success"):
                st.session_state.upload_status = None
                st.session_state.upload_result = None
                st.rerun()
    
    elif st.session_state.upload_status == "error" and st.session_state.upload_result:
        col1, col2 = st.columns([4, 1])
        with col1:
            error_msg = st.session_state.upload_result.get("error", "Unknown error")
            st.markdown(f'<div class="error-box">❌ <strong>Upload Failed:</strong> {error_msg}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("✖️ Dismiss", key="dismiss_error"):
                st.session_state.upload_status = None
                st.session_state.upload_result = None
                st.rerun()
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents to add to the knowledge base"
    )
    
    if uploaded_files:
        st.write(f"📋 **Selected {len(uploaded_files)} file(s):**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        # Create columns for better button placement
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                # Clear previous status
                st.session_state.upload_status = None
                st.session_state.upload_result = None
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📄 Preparing files for upload...")
                    progress_bar.progress(20)
                    
                    status_text.text("⬆️ Uploading documents to server...")
                    progress_bar.progress(40)
                    
                    # Perform the actual upload
                    result = upload_documents(uploaded_files)
                    progress_bar.progress(80)
                    
                    status_text.text("🔍 Building search indices...")
                    progress_bar.progress(100)
                    
                    # Store result and set status
                    if result and "error" not in result:
                        st.session_state.upload_status = "success"
                        st.session_state.upload_result = result
                        status_text.text("✅ Upload completed successfully!")
                    else:
                        st.session_state.upload_status = "error"
                        st.session_state.upload_result = result or {"error": "Upload failed"}
                        status_text.text("❌ Upload failed!")
                    
                    # Clean up progress indicators after a moment
                    time.sleep(2)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Rerun to show persistent status
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.upload_status = "error"
                    st.session_state.upload_result = {"error": f"Unexpected error: {str(e)}"}
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
    else:
        st.info("📤 Choose PDF files above to get started")

def upload_documents(files) -> Dict[str, Any]:
    """Upload documents to the system."""
    try:
        files_data = [("files", (file.name, file.getvalue(), "application/pdf")) for file in files]
        
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            files=files_data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.status_code} - {response.text}"}
            
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def system_overview_interface():
    """System overview and architecture with dynamic capabilities."""
    st.header("🔧 System Architecture & Features")
    
    # Get current system stats
    stats = get_system_stats()
    
    if stats:
        system_info = stats.get("system", {})
        capabilities = system_info.get("capabilities", {})
        
        # Architecture overview - now dynamic
        st.subheader("📐 Current Pipeline Architecture")
        
        pipeline_components = system_info.get("pipeline_components", [])
        if pipeline_components:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("**🏗️ Active Pipeline Components:**")
            for i, component in enumerate(pipeline_components, 1):
                st.markdown(f"{i}. {component}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No documents processed yet. Upload PDFs to see active pipeline components.")
        
        # Dynamic feature highlights based on actual capabilities
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Active Features")
            active_features = []
            
            if capabilities.get("hybrid_retrieval"):
                active_features.append("**Hybrid Search:** Semantic + keyword search active")
            if capabilities.get("semantic_search"):
                active_features.append("**Semantic Search:** Vector embeddings operational")
            if capabilities.get("keyword_search"):
                active_features.append("**Keyword Search:** BM25 index built")
            if capabilities.get("table_processing"):
                active_features.append("**Table Awareness:** Structured data detected")
            if capabilities.get("llm_processing"):
                active_features.append("**LLM Integration:** Mistral AI connected")
            
            # Always available features
            active_features.extend([
                "**Evidence Filtering:** Similarity threshold enforcement",
                "**Citation System:** Source tracking and confidence scoring"
            ])
            
            for feature in active_features:
                st.markdown(f"- {feature}")
        
        with col2:
            st.subheader("🛡️ Safety & Quality")
            safety_features = [
                "**PII Detection:** Personal information filtering",
                "**Content Filtering:** Medical/legal disclaimers",
                "**Evidence Thresholds:** Minimum similarity requirements",
                "**Hallucination Prevention:** Post-hoc fact checking",
                "**Query Refusal:** Harmful content policies",
                "**Transparency:** Complete processing metadata"
            ]
            
            for feature in safety_features:
                st.markdown(f"- {feature}")
        
        # Technical implementation - dynamic based on actual state
        st.subheader("⚙️ Current Technical Implementation")
        
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        
        with tech_col1:
            st.markdown("**🔍 Retrieval Engine:**")
            if capabilities.get("semantic_search"):
                embeddings_count = stats.get("retrieval", {}).get("embeddings_cached", 0)
                st.write(f"- Mistral embeddings ({embeddings_count} cached)")
            if capabilities.get("keyword_search"):
                terms_count = stats.get("retrieval", {}).get("keyword_index_terms", 0)
                st.write(f"- BM25 index ({terms_count} terms)")
            if capabilities.get("hybrid_retrieval"):
                st.write("- MMR re-ranking active")
            if not any(capabilities.values()):
                st.write("- Awaiting document ingestion")
        
        with tech_col2:
            st.markdown("**🧠 Language Model:**")
            if capabilities.get("llm_processing"):
                st.write("- Mistral AI integration active")
                st.write("- Intent detection operational")
                st.write("- Query transformation enabled")
                st.write("- Template-based generation")
            else:
                st.write("- LLM integration pending")
        
        with tech_col3:
            st.markdown("**📊 Data Processing:**")
            doc_count = stats.get("documents", {}).get("total_documents", 0)
            chunk_count = stats.get("documents", {}).get("total_chunks", 0)
            table_count = stats.get("table_awareness", {}).get("total_tables_detected", 0)
            
            if doc_count > 0:
                st.write(f"- {doc_count} documents processed")
                st.write(f"- {chunk_count} searchable chunks")
                if table_count > 0:
                    st.write(f"- {table_count} tables detected")
                st.write("- Metadata enrichment active")
            else:
                st.write("- No documents processed yet")
        
        # Performance metrics
        st.subheader("📊 Current System Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        docs = stats.get("documents", {})
        retrieval = stats.get("retrieval", {})
        tables = stats.get("table_awareness", {})
        
        with metrics_col1:
            st.metric("Documents Indexed", docs.get("total_documents", 0))
            
        with metrics_col2:
            st.metric("Searchable Chunks", docs.get("total_chunks", 0))
            
        with metrics_col3:
            st.metric("Embeddings Cached", retrieval.get("embeddings_cached", 0))
            
        with metrics_col4:
            st.metric("Tables Detected", tables.get("total_tables_detected", 0))
        
        # System status indicator
        system_status = system_info.get("status", "unknown")
        status_colors = {
            "operational": "🟢 Fully Operational",
            "processing": "🟡 Processing Documents", 
            "awaiting_documents": "🔴 Awaiting Documents"
        }
        
        st.subheader("🖥️ System Status")
        st.markdown(f"**Status:** {status_colors.get(system_status, '⚪ Unknown')}")
        
        # Advanced features list - now dynamic
        advanced_features = system_info.get("advanced_features", [])
        if advanced_features:
            with st.expander("🚀 Advanced Features Currently Active"):
                for feature in advanced_features:
                    st.write(f"• {feature}")
        
        # Raw system info for debugging
        with st.expander("🔧 Complete System Information"):
            st.json(stats)
    
    else:
        st.error("Unable to retrieve system statistics. Please check if the backend is running.")
        st.markdown("""
        **Expected System Architecture:**
        1. Document Ingestion with PDF processing
        2. Query Processing with LLM integration  
        3. Hybrid Retrieval (Semantic + Keyword)
        4. Generation with Citations
        5. Safety and Quality Controls
        """)

if __name__ == "__main__":
    main()