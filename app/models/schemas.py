from pydantic import BaseModel
from typing import List

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
