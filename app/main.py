from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_emotion import router as emotion_router
from app.api.routes_replay import router as replay_router
# from app.api.routes_healing import router as healing_router  # (Optional - if you add this later)

app = FastAPI(
    title="Rewind Emotion Assistant API",
    description="API for detecting emotion, extracting memory context, and generating AI reflections",
    version="1.0.0"
)

# Optional: allow frontend or external tools to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set allowed domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(emotion_router, prefix="/api")
app.include_router(replay_router, prefix="/api")
# app.include_router(healing_router, prefix="/api")  # (Optional)

@app.get("/")
def root():
    return {"message": "Welcome to the Rewind Emotion Assistant API 🚀"}
