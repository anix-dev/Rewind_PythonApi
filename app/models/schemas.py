from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TextRequest(BaseModel):
    text: str

class ReplayRequest(BaseModel):
    user_text: str
    mood: str
    longitude: float
    latitude: float
    events: List[str]
    context_tags: List[str]
    create_date: str


class MoodCreateRequest(BaseModel):
    user_text: str
    audio_file: Optional[str]
    mood: str
    ai_response: Optional[str]
    user: str  # User ID as a string
    is_shown: Optional[bool] = True
    longitude: Optional[float]
    latitude: Optional[float]
    events: Optional[List[str]] = []
    context_tags: Optional[List[str]] = []
    replay_opportunity_score: Optional[str] = "0.5"
    create_date: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ReplayCreateRequest(BaseModel):
    gem_response: str
    user_response: str
    replay_opportunity_score: Optional[str] = "0.5"
    context_tags: Optional[List[str]] = []
    location: Optional[str]
    user: str  # String, will be converted to ObjectId
    moods: str  # String, will be converted to ObjectId
    create_date: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updatedAt: Optional[datetime] = Field(default_factory=datetime.utcnow)
