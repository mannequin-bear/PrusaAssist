"""
PrusaAssist — FastAPI Backend
Main entry point for the application.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    force=True  # <--- This destroys any hidden log catchers and forces terminal output
)
app = FastAPI(title="PrusaAssist API", version="1.0.0")
# ── Standard Setup ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

app.include_router(router)