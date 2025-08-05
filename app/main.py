from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.chroma_client import chroma_client

from app.api.routes_emotion import router as emotion_router
from app.api.routes_replay import router as replay_router
# from app.api.routes_healing import router as healing_router  # (Optional - if you add this later)

from app.db.mongo_client import verify_connection

app = FastAPI(
    title="Rewind Emotion Assistant API",
    description="API for detecting emotion, extracting memory context, and generating AI reflections",
    version="1.0.0"
)

# Allow cross-origin requests (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    try:
        await verify_connection()
        
        # Optional check if Chroma is initialized
        if chroma_client is not None:
            try:
                # Just a test collection fetch to confirm it's working
                chroma_client.get_or_create_collection(name="test-connection")
                print("✅ ChromaDB is initialized and ready.")
            except Exception as e:
                print(f"❌ ChromaDB initialization failed: {e}")
        else:
            print("⚠️ ChromaDB client is not initialized.")

    except Exception as e:
        print(f"❌ MongoDB connection failed during startup: {e}")


# Register your API routes
app.include_router(emotion_router, prefix="/api")
app.include_router(replay_router, prefix="/api")
# app.include_router(healing_router, prefix="/api")  # (Optional)

@app.get("/")
def root():
    return {"message": "Welcome to the Rewind Emotion Assistant API 🚀"}
