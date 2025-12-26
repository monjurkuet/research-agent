import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "ministral-3:latest")
    EMBED_MODEL = "nomic-embed-text"
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Storage Paths
    KB_DIR = "/mnt/h/research-agent/knowledge_base"
    DB_DIR = "./chroma_db"
    
    # Research Parameters
    MAX_QUERIES = 2  # How many angles the agent researches