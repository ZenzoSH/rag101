# RAG Document Chat System

A Retrieval-Augmented Generation (RAG) system using Pinecone, Ollama, and Streamlit.

## Features
- 📄 Multi-document upload (PDF, TXT, MD, etc.)
- 🔍 Semantic search with vector embeddings
- 💬 Chat with your documents using local LLM
- 🎨 Beautiful Streamlit UI

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set up Pinecone:
   - Sign up at https://pinecone.io
   - Get your API key
   - Create `.streamlit/secrets.toml`:
```toml
   PINECONE_API_KEY = "your_key_here"
```

3. Install Ollama models:
```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

4. Run the app:
```bash
streamlit run main.py
```

last upadted 11 Jan 2026 at 9:47 PM
Sunday
9:47 PM


Still in progress..Testing and running right now.


## Usage
1. Upload documents via sidebar
2. Click "Process Documents"
3. Start chatting!

