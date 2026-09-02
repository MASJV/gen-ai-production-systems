# Enterprise Knowledge Assistant — Advanced RAG Capstone

An enterprise multi-document RAG (Retrieval-Augmented Generation) system.
Upload PDF / DOCX / TXT / MD / CSV files, ask a question, and get an
answer grounded in your own documents — with hybrid search and reranking.

## Demo

A short walkthrough of the app in action — upload documents, ask a
question, and see the grounded answer with its retrieved source chunks.

- **Live app:** https://enterprise-knowledge-assistant-jv.streamlit.app/
- **Demo video:** https://drive.google.com/file/d/1AYEjy70bPxON1RCLzgnhFa5jVh-ABDOs/view?usp=sharing
- **Slides:** [architecture-overview.pptx](./architecture-overview.pptx)

## Architecture

```
Streamlit Dashboard
       |
  Upload Files ---------------- Ask Question
       |
LlamaIndex Document Readers
       |
PDF / DOCX / TXT / MD / CSV Parsing
       |
    Chunking
       |
OpenAI Embeddings (text-embedding-3-small)
       |
   Chroma Vector Store
       |
   Hybrid Retrieval (Vector Search + BM25)
       |
   BGE Reranker (Free, local)
       |
   Top Relevant Chunks
       |
   OpenAI GPT-4.1 mini
       |
   Final Answer
```

## Files and which stage they handle

| File | Architecture stage |
|---|---|
| `document_loader.py` | LlamaIndex Document Readers + file parsing |
| `chunking.py` | Chunking |
| `embeddings_store.py` | OpenAI Embeddings + Chroma Vector Store |
| `hybrid_retriever.py` | Hybrid Retrieval (Vector + BM25) |
| `reranker.py` | BGE Reranker |
| `generator.py` | OpenAI GPT-4.1 mini -> Final Answer |
| `app.py` | Streamlit Dashboard (connects every stage) |
| `config.py` | All settings in one place |

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      (Windows)
   source venv/bin/activate   (Mac/Linux)
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Enter your OpenAI API key in the sidebar and click **Set**.
The key is stored only for the current Streamlit session.

## How it works, in plain words

1. **Upload** — You upload one or more documents in the browser.
2. **Read & Parse** — LlamaIndex opens each file, no matter the format,
   and pulls out plain text.
3. **Chunk** — Long text is cut into small overlapping pieces so the
   AI can search and read them efficiently.
4. **Embed & Store** — Each chunk is turned into a list of numbers
   (an embedding) using OpenAI's cheapest embedding model, and saved
   in Chroma, a free local vector database.
5. **Hybrid Retrieval** — When you ask a question, we search two ways:
   - **Vector search**: finds chunks with similar *meaning*.
   - **BM25**: finds chunks with matching *keywords*.
   Both lists are merged using Reciprocal Rank Fusion (RRF).
6. **Rerank** — A free local model (BGE reranker) looks closely at the
   question and each candidate chunk together, and picks the best few.
7. **Generate** — The best chunks are sent to OpenAI's `gpt-4.1-mini`
   along with your question, and it writes the final answer.

## Cost notes

This project only uses cheap OpenAI models:
- `text-embedding-3-small` for embeddings
- `gpt-4.1-mini` for the final answer

Chroma (vector database) and the BGE reranker both run for **free**
on your own computer — no extra API cost for search or reranking.