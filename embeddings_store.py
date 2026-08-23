import chromadb
from openai import OpenAI

import config


def get_embedding(text, client):
    """
    Sends one piece of text to OpenAI and gets back its embedding
    (a list of numbers).
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    embedding = response.data[0].embedding
    return embedding


def build_vector_store(chunks, api_key):
    """
    Creates a fresh Chroma collection and stores every chunk's
    embedding inside it. Returns the collection so we can search it later.
    """
    client = OpenAI(api_key=api_key)

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIRECTORY)

    # remove old collection if it exists, so each run starts clean
    existing_collections = chroma_client.list_collections()
    for existing_collection in existing_collections:
        if existing_collection.name == config.COLLECTION_NAME:
            chroma_client.delete_collection(config.COLLECTION_NAME)

    collection = chroma_client.create_collection(config.COLLECTION_NAME)

    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for chunk_info in chunks:
        embedding = get_embedding(chunk_info["text"], client)
        ids.append(chunk_info["id"])
        texts.append(chunk_info["text"])
        metadatas.append({"source": chunk_info["source"]})
        embeddings.append(embedding)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return collection


def vector_search(collection, question, top_k, api_key):
    """
    Finds the top_k chunks whose meaning is closest to the question.
    """
    client = OpenAI(api_key=api_key)

    question_embedding = get_embedding(question, client)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    matched_chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    for i in range(len(documents)):
        chunk_info = {
            "id": ids[i],
            "text": documents[i],
            "source": metadatas[i]["source"]
        }
        matched_chunks.append(chunk_info)

    return matched_chunks