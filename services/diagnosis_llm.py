"""
Diagnostic Synthesis Service
Takes the raw inputs, the RAG context, and the image to generate the final JSON.
"""

import base64
import json
import logging
import google.generativeai as genai
from pydantic import ValidationError
from core.config import settings
from api.schemas import DiagnosticResult

# Fixed typo in logger name
log = logging.getLogger("prusaassist.diagnostic_llm")
llm = genai.GenerativeModel(settings.GEMINI_PRO)

def generate_diagnosis(
    clean_query:   str,
    rag_context:   str,
    text_input:    str | None,
    image_bytes:   bytes | None,
    printer_model: str,
) -> DiagnosticResult:
    
    log.info("LLM: Generating diagnosis...")



    parts = []
    parts.append(f"""You are PrusaAssist, an expert maintenance AI for Prusa 3D printers.
Printer model: {settings.AVAILABLE_MODELS.get(printer_model, printer_model)}

{rag_context}

Respond in this EXACT JSON format:
{{
  "summary": "2-3 sentence concise summary of what the issue is and likely cause",
  "diagnostics": "Numbered step-by-step repair instructions. Format: 1. Step\\n2. Step",
  "references": ["Page X — description of what was found there"],
  "warnings": ["Safety warning text"]
}}

Rules:
- references: cite every page number used from the manual passages above
- warnings: ONLY real safety hazards (burns, electrical, fragile parts). Empty array if none.
- Return ONLY valid JSON. No markdown fences, no preamble.
- if the input feels irrelevant to the context, just tell that you cant answer""")

    if text_input:
        parts.append(f"\nTechnician's description: {text_input}")

    if image_bytes:
        mime = "image/jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime = "image/png"
        parts.append({
            "mime_type": mime,
            "data": base64.b64encode(image_bytes).decode(),
        })
        parts.append("The above image was taken by the technician. Use it to refine your diagnosis.")

    parts.append(f"\nPrimary issue identified: {clean_query}")

    # log.error(f"LLM: !CRITICAL! Gemini API call failed:")
    # demo_logs.append(f"[ERROR] LLM: !CRITICAL! Gemini API call failed:")
    # err += 1
    #     # Graceful fallback so the frontend gets a valid JSON response, not a server crash
    # return DiagnosticResult(
    #     summary="System encountered a network error communicating with the AI model.",
    #     diagnostics="Please check your internet connection or try again in a few moments.",
    #     references=[],
    #     warnings=[]
    # )

    # --- NEW: TRY/EXCEPT FOR THE NETWORK API CALL ---
    try:
        response = llm.generate_content(parts)
        raw      = response.text.strip()
    except Exception as e:
        log.error(f"LLM: !CRITICAL! Gemini API call failed: {e}")
        # Graceful fallback so the frontend gets a valid JSON response, not a server crash
        return DiagnosticResult(
            summary="System encountered a network error communicating with the AI model.",
            diagnostics="Please check your internet connection or try again in a few moments.",
            references=[],
            warnings=[]
        )

    # Strip markdown fences if model added them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    
    if raw.startswith("json"):
        raw = raw[4:].strip()

    # --- PYDANTIC VALIDATION (Already perfect) ---
    try:
        return DiagnosticResult.model_validate_json(raw)
    except (json.JSONDecodeError, ValidationError) as e:
        log.error(f"LLM: Output Validation failed: {e}\nRaw: {raw[:300]}")

        return DiagnosticResult(
            summary=raw[:500] + "...",
            diagnostics="Could not parse structured response from the AI. See summary.",
            references=[],
            warnings=[]
        )
    
