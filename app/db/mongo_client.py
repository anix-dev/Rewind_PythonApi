# app/db/mongo_client.py
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rewind")

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]

# Async ping check
async def verify_connection():
    try:
        await db.command("ping")
        print("✅ Connected to MongoDB successfully.")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

# Call it on module import via background task
asyncio.create_task(verify_connection())
