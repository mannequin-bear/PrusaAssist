import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()   

# Centralize the GenAI configuration here so it runs once upon import
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class Settings:
    CHROMA_PATH: str = "./data/chroma_db"
    SIGLIP_MODEL: str = "google/siglip-base-patch16-224"
    
    AVAILABLE_MODELS: dict[str, str] = {
        "prusa_mk2_5": "Prusa MK2.5",
        "prusa_mk4_s": "Prusa MK4S",
        "prusa_xl":    "Prusa XL",
    }

    GEMINI_LITE: str = "gemini-3.1-flash-lite"
    GEMINI_FLASH: str = "gemini-3.5-flash"
    GEMINI_PRO: str = "gemini-2.5-flash"

settings = Settings()
