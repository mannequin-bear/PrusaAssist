"""
RAG Engine Service
Handles local SigLIP embeddings, ChromaDB retrieval, and context formatting.
"""
import logging
import torch
from transformers import AutoModel, AutoProcessor
from core.config import settings

log = logging.getLogger("prusaassist.rag_engine")

# Load SigLIP once at startup
try:
    log.info("RAG_ENGINE: Loading SigLIP...")
    siglip_processor = AutoProcessor.from_pretrained(settings.SIGLIP_MODEL)
    siglip_model     = AutoModel.from_pretrained(settings.SIGLIP_MODEL)
    siglip_model.eval()
    log.info("RAG_ENGINE: SigLIP ready.")

except Exception as e:
    log.error(f"RAG_ENGINE: Failed to load SigLIP model. {e}")
    # We raise here because if the model fails at startup, the app shouldn't run.
    raise RuntimeError(f"Model initialization failed: {e}")

def embed_query(query: str) -> list[float]:
    """Generates a multimodal latent vector from the text query."""
    log.info("RAG_ENGINE: Embedding query...")
    
    try:
        inputs = siglip_processor(
            text=query,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        with torch.no_grad():
            features = siglip_model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()
        
    except Exception as e:
        log.error(f"RAG_ENGINE: Failed to embed query : {e}")
        # Return an empty list so the retrieve function knows to skip search
        return []

def retrieve_chunks(collection, query_embedding: list[float], n: int = 5) -> list[dict]:
    """Retrieves the top-n closest chunks from the ChromaDB collection."""
    if not query_embedding:
        log.warning("RAG_ENGINE: Empty query embedding received. Skipping retrieval.")
        return []

    log.info("RAG_ENGINE: Querying ChromaDB...")
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        
        chunks = []
        # Check if results exist before zipping to avoid index errors on empty collections
        if not results.get("documents") or not results["documents"][0]:
            log.warning("ChromaDB returned no results.")
            return chunks

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1.0 - dist, 4)
            # Threshold filter to ensure relevance
            if score > 0.25:
                chunks.append({
                    "text":   doc,
                    "page":   meta.get("page", "?"),
                    "source": meta.get("source", "manual"),
                    "type":   meta.get("type", "text"),
                    "score":  score,
                })
                
        log.info(f"RAG_ENGINE: Retrieved {len(chunks)} chunks (top score: {chunks[0]['score'] if chunks else 'N/A'})")
        return chunks
        
    except Exception as e:
        log.error(f"RAG_ENGINE: ChromaDB retrieval failed: {e}")
        return []

def build_rag_context(chunks: list[dict]) -> str:
    """Formats the raw database chunks into a string for the LLM prompt."""
    if not chunks:
        return "[No relevant manual sections found. Answer from general knowledge.]"
        
    lines = ["Retrieved from official Prusa documentation:\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[Passage {i} | Page {c['page']} | relevance {c['score']}]")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)