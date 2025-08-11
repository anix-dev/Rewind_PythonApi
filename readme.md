 
 git bash 
 ======================
 
 pip install -r requirements.txt
python -m venv venv

source venv/Scripts/activate

 uvicorn main:app --reload

python -m uvicorn app.main:app --reload



Rewind_PythonApi/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_emotion.py       # /analyze, /detect-mood/text
│   │   ├── routes_replay.py        # /replay
│   │   └── routes_healing.py       # /healing-reflection
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── emotion_analyzer.py     # HuggingFace emotion pipeline
│   │   ├── memory_extractor.py     # Keyword/tag extractor, NLP utils
│   │   ├── embedding_generator.py  # Instructor/MiniLM vectorizer
│   │   ├── llama_interface.py      # LLaMA 3 prompt wrapper via Ollama
│   │   └── rag_engine.py           # LlamaIndex + ChromaDB integration
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── mongo_models.py         # MongoDB schema (if using ODM)
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── chroma_client.py        # ChromaDB setup
│   │   └── mongo_client.py         # MongoDB connection + utils
│   │
│   ├── prompts/
│   │   └── healing_template.txt    # LLaMA prompt for reflection + healing
│   │
│   ├── main.py                     # FastAPI app entrypoint
│
├── .env                            # API keys, Mongo URI, model configs
├── requirements.txt                # Python dependencies
├── README.md
