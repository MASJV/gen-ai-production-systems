"""
reranker.py

Stage: "BGE Reranker (Free)" -> "Top Relevant Chunks"

Hybrid search gives us a decent list of chunks, but the order is not
always perfect. A reranker looks at the question and each chunk
TOGETHER (not separately, like embeddings do) and gives a more
accurate relevance score. This model runs on your own computer,
so there is no extra API cost.
"""

from sentence_transformers import CrossEncoder

import config

# this downloads the model once and keeps it in memory
reranker_model = CrossEncoder(config.RERANKER_MODEL)


def rerank_chunks(question, chunks, top_k):
    """
    Re-scores every chunk against the question using the BGE
    reranker model, then returns only the best top_k chunks.
    """
    pairs = []
    for chunk_info in chunks:
        pairs.append([question, chunk_info["text"]])

    scores = reranker_model.predict(pairs)

    scored_chunks = []
    for i in range(len(chunks)):
        scored_chunks.append((scores[i], chunks[i]))

    # highest score first
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)

    top_chunks = []
    for i in range(top_k):
        if i < len(scored_chunks):
            top_chunks.append(scored_chunks[i][1])

    return top_chunks
