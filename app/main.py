# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Database connections
from app.db.mongo_client import verify_connection
from app.db.chroma_client import chroma_client
from app.db import llama_index_client  # 🧠 triggers LlamaIndex initialization

# Routers
from app.api.routes_emotion import router as emotion_router
from app.api.routes_replay import router as replay_router
# from app.api.routes_healing import router as healing_router  # Optional: if you add later

# FastAPI App Initialization
app = FastAPI(
    title="Rewind Emotion Assistant API",
    description="API for detecting emotion, extracting memory context, and generating AI reflections",
    version="1.0.0"
)

# CORS Setup (Allow all origins for dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: Mongo + ChromaDB + LlamaIndex
@app.on_event("startup")
async def startup_event():
    try:
        await verify_connection()
        print("✅ MongoDB connection verified.")
    except Exception as e:
        print(f"❌ MongoDB connection failed during startup: {e}")

    if chroma_client:
        try:
            chroma_client.get_or_create_collection(name="test-connection")
            print("✅ ChromaDB is initialized and ready.")
        except Exception as e:
            print(f"❌ ChromaDB test connection failed: {e}")
    else:
        print("⚠️ ChromaDB client is not initialized.")

    if llama_index_client.index:
        print("✅ LlamaIndex is ready and integrated with ChromaDB.")
    else:
        print("❌ LlamaIndex is not initialized properly.")

# API Routers
app.include_router(emotion_router, prefix="/api")
app.include_router(replay_router, prefix="/api")
# app.include_router(healing_router, prefix="/api")  # Optional: if implemented

# Health Check
@app.get("/")
def root():
    return {"message": "Welcome to the Rewind Emotion Assistant API 🚀"}
