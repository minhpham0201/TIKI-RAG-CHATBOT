import os
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from functions.vector_store import chunk_and_store_document
from functions.helper import delete_chunks_by_prefix, list_all_chunks, retrieve_chunks

# Initialize ChromaDB client & collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("tiki_doc")
FOLDER_PATH = 'Policy'
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

def setup():
    """Initializes ChromaDB if empty, otherwise returns existing collection."""
    existing_chunks = collection.count()

    if existing_chunks == 0:  # Check if collection is empty
        print("Initializing vector store in ChromaDB...")

        if os.path.isdir(FOLDER_PATH):  # Ensure FOLDER_PATH is a valid directory
            for filename in os.listdir(FOLDER_PATH):
                file_path = os.path.join(FOLDER_PATH, filename)

                if os.path.isfile(file_path):  # Ensure it's a file, not a folder
                    print(f"Processing: {file_path}")
                    chunk_and_store_document(file_path=file_path, embedding_model=EMBEDDING_MODEL)
        else:
            print(f"❌ ERROR: {FOLDER_PATH} is not a valid directory or does not exist.")

    else:
        print(f"✅ Vector store already initialized with {existing_chunks} chunks.")

    return collection  # Return collection for further use

def main():
    setup()  # Ensure vector store is initialized
    print("Listing all stored chunks:")



if __name__ == '__main__':  # Fixed conditional check
    main()


