import os
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter



def chunk_and_store_document(file_path, embedding_model):
    """Reads a document, splits into chunks, embeds, and stores them in ChromaDB."""
   
    # Initialize ChromaDB client & collection
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_client.delete_collection("tiki_doc")
    collection = chroma_client.get_or_create_collection("tiki_doc")

    # Load embedding model
    model = embedding_model

    doc_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract document name

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Text chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "], keep_separator=False
    )
    chunks = text_splitter.create_documents([text])  # Split into chunks

    # Embed and store each chunk separately
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        embedding = model.encode(chunk_text).tolist()

        chunk_id = f"{doc_name}_{idx}"  # Unique ID for each chunk
        metadata = {
            "document_name": doc_name,
            "filename": file_path,
            "chunk_id": idx,
            "created_at": datetime.utcnow().isoformat(),  # Store timestamp
        }

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk_text],
            metadatas=[metadata]  # Storing metadata only
        )

    print(f"✅ Stored {len(chunks)} chunks from '{file_path}' in ChromaDB!")

    return 

