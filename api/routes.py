"""
API Routes
Defines the endpoints for the PrusaAssist backend.
"""

import time
import logging
import chromadb
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from core.config import settings
from api.schemas import AnalyzeResponse, DiagnosticResult
# Import our extracted cognitive services
from services.vision_agent import analyze_image_defect
from services.audio_agent import transcribe_audio
from services.router_agent import rewrite_query
from services.rag_engine import embed_query, retrieve_chunks, build_rag_context
from services.diagnosis_llm import generate_diagnosis

# Set up module-specific logging

log = logging.getLogger("prusaassist.routes")
# Initialize the router
router = APIRouter()
# ── ChromaDB Client ───────────────────────────────────────────────────────────
# We initialize the client here so the routes can access the vector database.
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
def get_collection(printer_model: str):
    try:
        return chroma_client.get_collection(printer_model)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No knowledge base found for '{printer_model}'. Run ingest.py first.",
        )
# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Health check endpoint. Verifies ChromaDB collections are loaded."""
    collections = []
    for model_key in settings.AVAILABLE_MODELS:
        try:
            col = chroma_client.get_collection(model_key)
            collections.append({"name": model_key, "chunks": col.count(), "status": "ok"})
        except Exception:
            collections.append({"name": model_key, "chunks": 0, "status": "not_ingested"})
    return {"status": "online", "siglip": "loaded", "collections": collections}

@router.get("/models")
def list_models():
    """Returns the list of supported Prusa 3D printer models."""
    return {"models": settings.AVAILABLE_MODELS}

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    printer_model: str             = Form(...),
    question:      str | None      = Form(None),
    image:         UploadFile | None = File(None),
    audio:         UploadFile | None = File(None),
):
    """
    Main multimodal inference pipeline. 
    Accepts text, image, and audio inputs to diagnose a 3D printer issue.
    """
    t_start = time.time()
    log.info(f"--- Request | model={printer_model} | text={bool(question)} | "
             f"image={bool(image)} | audio={bool(audio)}")

    if printer_model not in settings.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown printer model: {printer_model}")

    collection  = get_collection(printer_model)
    image_bytes = await image.read() if image else None
    audio_bytes = await audio.read() if audio else None

    # Step 1: Speech-to-Text
    transcript = None
    if audio_bytes:
        try:
            mime = audio.content_type or "audio/webm"
            transcript = transcribe_audio(audio_bytes, mime_type=mime)
        except Exception as e:
            log.warning(f"STT failed: {e}")

    text_combined = " ".join(filter(None, [question, transcript])) or None

    # Step 2: Image description (Defect Detection)
    image_description = None
    if image_bytes:
        try:
            image_description = analyze_image_defect(image_bytes)
        except Exception as e:
            log.warning(f"Image description failed: {e}")

    # Step 3: Context Fusion & Query Rewriting
    clean_query = rewrite_query(image_description, text_combined)

    # Step 4: Multimodal Retrieval (Local SigLIP + ChromaDB)
    query_embedding = embed_query(clean_query)
    chunks          = retrieve_chunks(collection, query_embedding, n=5)
    rag_context     = build_rag_context(chunks)

    # Step 5: Synthesis & Reasoning (Diagnostic Generation)
    result: DiagnosticResult = generate_diagnosis(
        clean_query   = clean_query,
        rag_context   = rag_context,
        text_input    = text_combined,
        image_bytes   = image_bytes,
        printer_model = printer_model,
    )

    log.info(f"Done in {round(time.time() - t_start, 2)}s")
    # --- Structured Pydantic Response ---
    return AnalyzeResponse(
        summary     = result.summary,
        diagnostics = result.diagnostics,
        references  = result.references,
        warnings    = result.warnings,
        query_used  = clean_query,
        chunks_used = len(chunks),
    )