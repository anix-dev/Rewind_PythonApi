from fastapi import APIRouter, Query
from app.services.indexing_service import index_user_data
from app.db.mongo_client import db
from app.db import llama_index_client  # ✅ Import the client

router = APIRouter()

@router.post("/index-user-data")
async def index_user(user_id: str = Query(...)):
    moods = await db.moods.find({"user": user_id}).to_list(100)
    replays = await db.replays.find({"user": user_id}).to_list(100)

    await index_user_data(user_id, moods, replays)
    return {"message": f"Indexed data for user {user_id}"}

@router.get("/search-memories")
async def search_memories(user_id: str, query: str):
    response = llama_index_client.index.as_query_engine().query(query)
    return {"query": query, "results": str(response)}
