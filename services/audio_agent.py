"""
Audio Agent Service
"""
import base64
import logging
import google.generativeai as genai
from core.config import settings


# Set up module-specific logging
log = logging.getLogger("prusaassist.audio")
# Instantiate the model specific to this service
audio_model = genai.GenerativeModel(settings.GEMINI_FLASH)

def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    log.info("Audio Agent: Transcribing audio...")
    if not audio_bytes:
        log.warning("Audio Agent: No audio bytes provided!")
        return ""
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    try:
        response  = audio_model.generate_content([
            {"mime_type": mime_type, "data": audio_b64},
            "Transcribe this audio exactly. Return only the transcript, no commentary.",
        ])
        transcript = response.text.strip()
        log.info(f"Audio Agent: Transcription done")
        return transcript
    except Exception as e:
        log.error(f"Audio Agent: Transcription failed: {e}")
        raise RuntimeError(f"Audio processing failed: {str(e)}")
