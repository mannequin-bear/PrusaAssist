import fitz  # PyMuPDF — unchanged
from transformers import AutoProcessor, AutoModel  # SigLIP (replaces CLIPProcessor, CLIPModel)
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import torch
import numpy as np
import chromadb                          # replaces FAISS
import google.generativeai as genai      # replaces openai
import os
import base64
import io
import hashlib
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── SigLIP setup  ──────────────────────────────────────────────
MODEL_NAME = "google/siglip-base-patch16-224"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print(f"SigLIP loaded: {MODEL_NAME}")
print(f"Embedding dimension: {model.config.vision_config.hidden_size}")

# ── Embedding functions (same interface as CLIP, just different model) ─────────

def embed_image(image_data):
    """Embed image using SigLIP. Accepts file path or PIL Image."""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data.convert("RGB")
    #returning in the format of pytorch tensors
    inputs = processor(images=image, return_tensors="pt")

    #normalisation to unit vector 
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True) 

    return features.squeeze().tolist()


def embed_text(text: str):
    """Embed text using SigLIP. Same vector space as embed_image."""
    inputs = processor(
        text=text,
        return_tensors="pt",
        padding="max_length",
        max_length=64,          # SigLIP uses 64, CLIP used 77
        truncation=True
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().tolist()

# ── ChromaDB setup ──────────────────────────────────────────

CHROMA_PATH     = "./data/chroma_db"
EMBED_DIM       = 768   # SigLIP base hidden size

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ── Hierarchical Collections Mapping ─────────────────────────
PRINTER_MANUALS = {
    "prusa_mk2_5": "./data/mk2.5.pdf",
    "prusa_mk4_s": "./data/mk4_s.pdf",
    "prusa_xl": "./data/xl.pdf"
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]  # priority order
)

def simple_chunk(text: str) -> list[str]:
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]

def ingest_pdf_to_collection(pdf_path: str, collection_name: str):
    if not os.path.exists(pdf_path):
        print(f"Warning: {pdf_path} not found. Skipping {collection_name}.")
        return

    # Delete and recreate for fresh ingestion
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"\n--- Ingesting to collection '{collection_name}' ---")
    
    doc = fitz.open(pdf_path)
    text_count  = 0
    image_count = 0

    print(f"Processing {len(doc)} pages from {pdf_path}...")

    for page_num, page in enumerate(doc):
        # ── TEXT ──
        print(f"Processing page number {page_num}")
        text = page.get_text()
        if text.strip():
            chunks = simple_chunk(text)
            for chunk_idx, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                chunk_id  = hashlib.md5(f"text:{collection_name}:{page_num}:{chunk_idx}".encode('utf-8')).hexdigest()
                collection.upsert(
                    ids        = [chunk_id],
                    embeddings = [embedding],
                    documents  = [chunk],
                    metadatas  = [{"type": "text", "page": page_num, "source": pdf_path}],
                )
                text_count += 1

        # ── IMAGES ──
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref        = img[0]
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # PIL image for SigLIP embedding
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Base64 for Gemini vision
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                image_id  = f"img_{collection_name}_{page_num}_{img_index}"

                # Embed image with SigLIP
                embedding = embed_image(pil_image)
                collection.upsert(
                    ids        = [image_id],
                    embeddings = [embedding],
                    documents  = [f"[Image on page {page_num} of {pdf_path}]"],
                    metadatas  = [{"type": "image", "page": page_num, "image_id": image_id, "source": pdf_path, "b64": b64_image, "mime": base_image.get("ext", "png")}],
                )
                image_count += 1

            except Exception as e:
                print(f"  Skipped image on page {page_num}: {e}")

    print(f"Collection '{collection_name}' ingestion complete!")
    print(f"  Text chunks : {text_count}")
    print(f"  Images      : {image_count}")
    print(f"  Total in DB : {collection.count()}")

if __name__ == "__main__":
    for coll_name, pdf_file in PRINTER_MANUALS.items():
        ingest_pdf_to_collection(pdf_file, coll_name)