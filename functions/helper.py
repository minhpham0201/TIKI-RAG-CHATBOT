

def list_all_chunks(collection):
    """Lists all stored chunks in ChromaDB with their full details."""
    results = collection.get(include=["documents", "embeddings", "metadatas"])  # Fetch everything

    if not results.get("ids"):
        print("No chunks found in ChromaDB.")
        return

    print(f"📄 Found {len(results['ids'])} chunks in ChromaDB:\n" + "-" * 50)

    for i, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i] if results.get("metadatas") and results["metadatas"] else {}

        print(f"🔹 Chunk ID: {chunk_id}")
        print(f"📑 Document Name: {metadata.get('filename', 'Unknown Document')}")


        print(f"🗂 Metadata: {metadata}")
        print("-" * 50)



def delete_chunks_by_prefix(collection,prefix):
    """Deletes all chunks from ChromaDB where the ID starts with the given prefix."""
    # Retrieve all stored metadata (including IDs)
    results = collection.get(include=["metadatas"])

    # Extract IDs that match the prefix
    chunk_ids_to_delete = [
        chunk_id for chunk_id in results["ids"] if chunk_id.startswith(prefix)
    ]

    # Delete matching chunks
    if chunk_ids_to_delete:
        collection.delete(ids=chunk_ids_to_delete)
        print(f"Deleted {len(chunk_ids_to_delete)} chunks with prefix '{prefix}' from ChromaDB.")
    else:
        print(f"No chunks found with prefix '{prefix}'.")




def retrieve_chunks(collection,query, top_k=5):
    """Retrieves the top_k most relevant chunks from ChromaDB based on the query."""

    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    # Search for relevant chunks in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],  # Search using embeddings
        n_results=top_k,  # Number of relevant chunks to retrieve
        include=["metadatas", "documents"]  # Include metadata and text
    )

    # Check if any documents were retrieved
    if not results.get("documents") or not results["documents"][0]:
        print("🚫 No relevant chunks found.")
        return []

    retrieved_chunks = []
    print(f"\n🔍 **Top {top_k} Relevant Chunks:**")
    print("-" * 50)

    # Iterate through retrieved chunks
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results.get("metadatas", [[]])[0])):
        metadata = metadata or {}  # Ensure metadata is a dictionary
        doc_name = metadata.get("document_name", "Unknown Document")
        text_preview = doc[:200] if doc else "[No Text Available]"

        print(f"🔹 {i+1}. {doc_name}")
        print(f"📜 Text Preview: {text_preview}...")
        print(f"🗂 Metadata: {metadata}")
        print("-" * 50)

        retrieved_chunks.append({"text": doc, "metadata": metadata})

    return retrieved_chunks
