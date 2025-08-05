from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore



from app.db.embedding_model import embed_model   # correct
from app.db.chroma_client import chroma_client


async def index_user_data(user_id: str, moods: list, replays: list):
    """
    Combine moods and AI replays into Document objects and index them into ChromaDB.
    """
    texts = []

    # Format moods
    for mood in moods:
        text = mood.get("user_text", "")
        mood_tag = mood.get("mood", "")
        if text:
            texts.append(f"[MOOD] {text} (Mood: {mood_tag})")

    # Format AI replays
    for replay in replays:
        ai_response = replay.get("ai_response", "")
        if ai_response:
            texts.append(f"[REPLAY] {ai_response}")

    # Create LlamaIndex documents with metadata
    documents = [Document(text=t, metadata={"user_id": user_id}) for t in texts]

    # Create or reuse ChromaDB collection
    collection = chroma_client.get_or_create_collection(name="memories")

    # Wrap with LlamaIndex's ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index documents (replaces insert_documents)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    print(f"✅ Indexed {len(documents)} documents for user_id: {user_id}")
