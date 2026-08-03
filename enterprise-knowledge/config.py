"""
config.py

This file keeps all the settings in one place.
If you want to change a model name or a number later,
you only need to change it here, not in every file.
"""

# No API key here because it is entered in the Streamlit sidebar per session.

# --- Models (cheapest OpenAI models only, per stack rule) ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# --- Free local reranker model (no API cost) ---
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Chunking settings ---
CHUNK_SIZE = 800      # number of characters in each chunk
CHUNK_OVERLAP = 100   # how many characters overlap between chunks

# --- Retrieval settings ---
TOP_K_VECTOR = 10   # how many chunks to get from vector search
TOP_K_BM25 = 10      # how many chunks to get from keyword search
TOP_K_FINAL = 4      # how many chunks to keep after reranking

# --- Storage settings ---
CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "enterprise_documents"
UPLOAD_FOLDER = "uploaded_docs"
