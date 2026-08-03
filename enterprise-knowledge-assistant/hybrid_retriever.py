"""
hybrid_retriever.py

Stage: "Hybrid Retrieval (Vector Search + BM25 Retriever)"

Vector search is great at understanding meaning, but it can miss
exact keywords (like product codes or names). BM25 is a classic
keyword-matching algorithm that is great at exact words.

Using both together and combining ("fusing") their results usually
gives better answers than using just one method alone.
"""

from rank_bm25 import BM25Okapi
import config
import embeddings_store


def build_bm25_index(chunks):
    """
    Builds a BM25 keyword search index from all the chunks.
    """
    tokenized_texts = []
    for chunk_info in chunks:
        words = chunk_info["text"].lower().split()
        tokenized_texts.append(words)

    bm25_index = BM25Okapi(tokenized_texts)
    return bm25_index


def bm25_search(bm25_index, chunks, question, top_k):
    """
    Finds the top_k chunks that best match the question's keywords.
    """
    question_words = question.lower().split()
    scores = bm25_index.get_scores(question_words)

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


def fuse_results(vector_chunks, bm25_chunks):
    """
    Combines the vector search list and the BM25 list into one list,
    using a simple method called Reciprocal Rank Fusion (RRF).

    The idea: a chunk that appears near the top of either list gets
    a high score. A chunk that appears near the top of BOTH lists
    gets an even higher score.
    """
    combined_scores = {}
    all_chunks_by_id = {}

    for rank, chunk_info in enumerate(vector_chunks):
        chunk_id = chunk_info["id"]
        score = 1 / (rank + 1)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + score
        all_chunks_by_id[chunk_id] = chunk_info

    for rank, chunk_info in enumerate(bm25_chunks):
        chunk_id = chunk_info["id"]
        score = 1 / (rank + 1)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + score
        all_chunks_by_id[chunk_id] = chunk_info

    sorted_ids = sorted(
        combined_scores.keys(),
        key=lambda chunk_id: combined_scores[chunk_id],
        reverse=True
    )

    fused_chunks = []
    for chunk_id in sorted_ids:
        fused_chunks.append(all_chunks_by_id[chunk_id])

    return fused_chunks


def hybrid_search(collection, bm25_index, chunks, question, api_key):
    """
    Runs vector search and BM25 search separately, then fuses
    the two result lists into one ranked list of chunks.
    """
    vector_chunks = embeddings_store.vector_search(
        collection, question, config.TOP_K_VECTOR, api_key
    )
    bm25_chunks = bm25_search(
        bm25_index, chunks, question, config.TOP_K_BM25
    )

    fused_chunks = fuse_results(vector_chunks, bm25_chunks)
    return fused_chunks
