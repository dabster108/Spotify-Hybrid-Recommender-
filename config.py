# config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
