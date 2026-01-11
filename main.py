import streamlit as st
import ollama
from pinecone import Pinecone, ServerlessSpec
import time
import os

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide"
)

# ============================================
# CONFIGURATION
# ============================================

PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "pcsk_2Awb6K_7PVDP5g1iAWoNx61reyGbDNXWWHdvcqqeCJnCE9xeWQC8SjNV48p5WE78nC53FZ")
INDEX_NAME = 'streamlit-rag'
DIMENSION = 768  # BGE-base model produces 768-dim embeddings
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# ============================================
# SESSION STATE
# ============================================

if 'index' not in st.session_state:
    st.session_state.index = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False

# ============================================
# FUNCTIONS
# ============================================

@st.cache_resource
def setup_pinecone():
    """Initialize Pinecone (cached)"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(1)
    
    return pc.Index(INDEX_NAME)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < text_length:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]

def process_uploaded_file(uploaded_file, index):
    """Process a single uploaded file"""
    try:
        # Read file content based on type
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            from pypdf import PdfReader
            from io import BytesIO
            
            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
            content = ''
            for page in pdf_reader.pages:
                content += page.extract_text() + '\n'
        else:
            content = uploaded_file.read().decode('utf-8')
        
        # Split into chunks
        chunks = chunk_text(content)
        
        if not chunks:
            st.warning(f'No content extracted from {uploaded_file.name}')
            return 0
        
        # Create embeddings and upload
        vectors = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            # Update progress
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            status_text.text(f'Processing chunk {i+1}/{len(chunks)}...')
            
            # Get embedding
            embedding = ollama.embed(
                model=EMBEDDING_MODEL,
                input=chunk
            )['embeddings'][0]
            
            # Prepare vector
            vector = {
                'id': f'{uploaded_file.name}_{i}',
                'values': embedding,
                'metadata': {
                    'text': chunk,
                    'source': uploaded_file.name,
                    'chunk_index': i
                }
            }
            vectors.append(vector)
        
        # Upload to Pinecone
        if vectors:
            index.upsert(vectors=vectors)
        
        progress_bar.empty()
        status_text.empty()
        
        return len(chunks)
        
    except Exception as e:
        st.error(f'Error processing {uploaded_file.name}: {e}')
        return 0

def retrieve(query, index, top_n=3):
    """Retrieve similar chunks from Pinecone"""
    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]
    
    results = index.query(
        vector=query_embedding,
        top_k=top_n,
        include_metadata=True
    )
    
    retrieved = []
    for match in results['matches']:
        text = match['metadata']['text']
        source = match['metadata'].get('source', 'unknown')
        score = match['score']
        retrieved.append((text, source, score))
    
    return retrieved

def generate_response(query, index):
    """Generate response using RAG"""
    retrieved = retrieve(query, index)
    
    # Build context
    context = '\n'.join([f'[{source}] {text}' for text, source, _ in retrieved])
    
    instruction_prompt = f'''You are a helpful assistant.
Use only the following context to answer the question. Don't make up information:
{context}
'''
    
    # Generate response
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query},
        ],
        stream=True,
    )
    
    # Stream response
    response = ""
    message_placeholder = st.empty()
    
    for chunk in stream:
        response += chunk['message']['content']
        message_placeholder.markdown(response + "▌")
    
    message_placeholder.markdown(response)
    
    return response, retrieved

# ============================================
# UI
# ============================================

st.title("📚 RAG Document Chat with Pinecone")
st.markdown("Upload documents and chat with them using AI")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    
    # Initialize Pinecone
    if st.session_state.index is None:
        with st.spinner("Connecting to Pinecone..."):
            st.session_state.index = setup_pinecone()
        st.success("✓ Connected to Pinecone")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['txt', 'md', 'py', 'json', 'csv', 'pdf'],
        accept_multiple_files=True,
        help="Upload text files or PDFs to chat with"
    )
    
    if uploaded_files and st.button("Process Documents", type="primary"):
        total_chunks = 0
        
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                st.write(f"📄 {uploaded_file.name}")
                chunks = process_uploaded_file(uploaded_file, st.session_state.index)
                total_chunks += chunks
                st.success(f"✓ Processed {chunks} chunks")
        
        st.success(f"🎉 All done! Processed {total_chunks} total chunks")
        st.session_state.documents_uploaded = True
    
    # Stats
    st.divider()
    st.subheader("📊 Stats")
    try:
        stats = st.session_state.index.describe_index_stats()
        st.metric("Total Vectors", stats['total_vector_count'])
    except:
        st.metric("Total Vectors", "N/A")
    
    # Clear database
    if st.button("🗑️ Clear Database", help="Delete all vectors"):
        if st.session_state.index:
            st.session_state.index.delete(delete_all=True)
            st.session_state.messages = []
            st.success("✓ Database cleared")
            st.rerun()

# Main chat interface
st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for text, source, score in message["sources"]:
                    st.markdown(f"**[{source}]** (similarity: {score:.2f})")
                    st.text(text[:200] + "...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.documents_uploaded:
        st.warning("⚠️ Please upload and process documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response, sources = generate_response(prompt, st.session_state.index)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

# Instructions
if not st.session_state.documents_uploaded:
    st.info("""
    👈 **Get Started:**
    1. Upload documents in the sidebar
    2. Click "Process Documents"
    3. Start chatting!
    """)