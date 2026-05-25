"""
Vision Agent Service
Handles all image processing and defect detection using Gemini 2.5 Flash.
"""

import base64
import logging
import google.generativeai as genai
from core.config import settings

# Set up module-specific logging
log = logging.getLogger("prusaassist.vision")

# Instantiate the model specific to this service
vision_model = genai.GenerativeModel(settings.GEMINI_FLASH)

def analyze_image_defect(image_bytes: bytes) -> str:
    log.info("Vision Agent: Analyzing image...")
    if not image_bytes:
        log.warning("Vision Agent: No image bytes provided!")
       
        return ""

    mime = "image/jpeg"
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif image_bytes[:4] == b'RIFF':
        mime = "image/webp"

    prompt = (
        "You are a strict 3D printer QA expert. Look at this image and describe:\n"
        "1. IMPORTANT: The specific part or component shown neatly (e.g., 'Z-axis trapezoidal nut', 'extruder idler gear').\n"
        "2. Any visible damage, wear, stringing, layer shifting, or misalignment.\n"
        "3. Any visible error codes on screens.\n"
        "Be concise and highly technical. Maximum 5 sentences."
    )

    try:
        response = vision_model.generate_content([
            {"mime_type": mime, "data": base64.b64encode(image_bytes).decode("utf-8")},
            prompt,
        ])
        
        description = response.text.strip()
        log.info(f"Vision Agent: Extraction Done")
        return description
        
    except Exception as e:
        log.error(f"Vision Agent: Failed to process image: {e}")
        
        # In a hackathon, we don't want the whole server to crash if the API hiccups.
        # We raise a controlled error that the main router can catch and handle gracefully.
        raise RuntimeError(f"Vision processing failed: {str(e)}")