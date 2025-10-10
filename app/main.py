from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Trigger loading of models and clients
from app.db import embedding_model  # Loads embedding model
from app.db.chroma_client import chroma_client
from app.db import llama_index_client  # Initializes LlamaIndex
from app.db.mongo_client import verify_connection  # MongoDB check

# Routers
from app.api.routes_emotion import router as emotion_router
from app.api.routes_replay import router as replay_router
# from app.api.routes_healing import router as healing_router  # Optional
from app.api.routes_transcribe import router as transcribe_router

from app.api.routes_index import router as index_router
from app.api.upload_router import router as upload_router



app = FastAPI(
    title="Rewind Emotion Assistant API",
    description="API for detecting emotion, extracting memory context, and generating AI reflections",
    version="1.0.0",
    root_path="/ai"  # <--- Add this line
)


# CORS (Adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup Checks
@app.on_event("startup")
async def startup_event():
    # ✅ MongoDB
    try:
        await verify_connection()
        print("✅ MongoDB connection verified.")
    except Exception as e:
        print(f"❌ MongoDB connection failed during startup: {e}")

    # ✅ ChromaDB
    if chroma_client:
        try:
            chroma_client.get_or_create_collection(name="test-connection")
            print("✅ ChromaDB is initialized and ready.")
        except Exception as e:
            print(f"❌ ChromaDB test connection failed: {e}")
    else:
        print("⚠️ ChromaDB client is not initialized.")

    # ✅ Embedding model
    try:
        if embedding_model.embedder:
            print("✅ Embedding model loaded successfully.")
        else:
            print("❌ Embedding model failed to load.")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")

    # ✅ LlamaIndex
    try:
        if llama_index_client.index:
            print("✅ LlamaIndex is ready and integrated with ChromaDB.")
        else:
            print("❌ LlamaIndex is not initialized properly.")
    except Exception as e:
        print(f"❌ LlamaIndex error: {e}")


# Include API routes
app.include_router(emotion_router, prefix="/api")
app.include_router(replay_router, prefix="/api")
app.include_router(index_router, prefix="/api")
app.include_router(transcribe_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

# app.include_router(healing_router, prefix="/api")  # Optional

# Health Check Endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Rewind Emotion Assistant API 🚀"}
