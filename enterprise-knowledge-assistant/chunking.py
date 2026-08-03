def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def chunk_documents(documents, chunk_size, overlap):  # chunks each document separately and combines the results
    """
    Takes a list of LlamaIndex Document objects and turns them into
    a list of simple dictionaries, one dictionary per chunk:
        {
            "id": "chunk_0",
            "text": "...",
            "source": "myfile.pdf"
        }
    """

    all_chunks = []
    chunk_number = 0

    for document in documents:
        source_name = document.metadata.get("file_name", "unknown") 
        text_chunks = chunk_text(document.text, chunk_size, overlap)

        for text_chunk in text_chunks:
            chunk_id = "chunk_" + str(chunk_number)
            chunk_info = {
                "id": chunk_id,
                "text": text_chunk,
                "source": source_name
            }
            all_chunks.append(chunk_info)
            chunk_number = chunk_number + 1

    return all_chunks