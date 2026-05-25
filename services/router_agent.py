import logging
import google.generativeai as genai
from core.config import settings

log = logging.getLogger("prusaassist.routing")
router_model = genai.GenerativeModel(settings.GEMINI_LITE)

def rewrite_query(image_desc: str | None, text_input: str | None) -> str:
    log.info("Router Agent: Rewriting query....")


    if image_desc and text_input:
        combined = f"Visual: {image_desc}\nTechnician says: {text_input}"
    elif image_desc:
        combined = f"Visual: {image_desc}"
    else:
        combined = text_input or "general printer issue"
    
    prompt = f"""You are a query rewriter for a 3D printer maintenance RAG system.
Extract the core technical issue as a short search query (max 8 words).
Prioritize component names and failure modes.
Return ONLY the query, nothing else.

CRITICAL RULE: If the input is purely conversational noise, contains profanity, or lacks technical substance, return exactly: "general maintenance guidelines"

Input: {combined}
Query:"""

    try:
        response = router_model.generate_content(prompt)
        query = response.text.strip().strip('"').strip("'")
        log.info(f"Router Agent: Rewriting Succesfull")
        return query
        
    except Exception as e:
        log.error(f"Router Agent: Failed {e}")
        raise RuntimeError(f"Router failed: {str(e)}")


    
