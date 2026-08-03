"""
document_loader.py

Stage: "LlamaIndex Document Readers" + "PDF / DOCX / TXT / MD / CSV Parsing"

This file has two simple jobs:
1. Save the files the user uploaded on Streamlit to a folder on disk.
2. Use LlamaIndex to read those files and turn them into plain text
   documents, no matter if they are PDF, DOCX, TXT, MD, or CSV.
"""
import os
from llama_index.core import SimpleDirectoryReader 

def save_uploaded_files(uploaded_files, folder_path):
    """
    Saves files coming from the Streamlit file uploader
    into a normal folder on disk, so LlamaIndex can read them.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # remove old files first, so we do not mix old and new uploads
    for old_file_name in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, old_file_name)
        if os.path.isfile(old_file_path):
            os.remove(old_file_path)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as output_file:
            output_file.write(uploaded_file.getbuffer())

    return folder_path

def load_documents(folder_path):
    """
    Reads every supported file inside folder path and returns a list of
    llamaIndex Document objects. Each document has.text and .metadata
    """
    reader = SimpleDirectoryReader(input_dir=folder_path) # only reads data
    documents = reader.load_data() # extracts text and metadata from the data
    return documents