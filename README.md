# PrusaAssist

PrusaAssist is a production-ready, multimodal RAG backend designed for hardware maintenance technicians. By combining local multimodal retrieval (ChromaDB + SigLIP) with a tiered agentic workflow (Gemini Flash & Pro), the system ingests images, audio, and text to diagnose 3D printer defects. It bypasses vague user descriptions by visually grounding the problem, querying official manuals, and returning structured, step-by-step repair instructions with strict API contracts and request-isolated observability.

## 1. System Overview:
PrusaAssist is a fully operational, end-to-end multimodal AI system engineered to provide actionable guidance to maintenance technicians. Having the capability of processing audio, image and text input formats bypasses the limitations of a generic chatbot. The system bases its diagnostic reasoning strictly within technical manuals, delivering safe, sequential repair instructions formatted specifically. The system uses lite, flash, pro AI Models.
The system is intentionally designed around the three core requirements: a real serving layer (using FastAPI), a knowledge layer (ChromaDB + SigLIP RAG), and a genuinely multimodal interface (image + voice + text). The chosen domain; Prusa 3D printer maintenance - offers publicly available technical manuals and a realistic population of failure modes that helped test the retrieval quality. Lite - gemini 3.1 flash lite, Flash - gemini 3.5 flash, Pro - gemini 3.5 flash.

## 2. Architecture & System Design
The system follows a clean, layered architecture with strict separation of concerns:

Ingestion: Offline, uses PyMuPDF to extract both text and embedded images from each printer manual. Text and images are embedded with a SigLIP encoder. Both land in the same vector space, enabling true multi-modal retrieval.

Serving Layer: The frontend (a single-page vanilla HTML/JS application) communicates exclusively through a FormData POST to the /analyze endpoint. The API layer (FastAPI + Pydantic schemas) validates and deserializes inputs, then hands off to a five-stage service pipeline. All AI model interactions are isolated into dedicated service modules, and the knowledge base is a persistent ChromaDB instance populated offline via a separate ingest script.

Audio Agent: Uses Flash, performs STT transcription.
Vision Agent: Flash performs a technically grounded visual analysis to identify components, defect, wear patterns to give a 5-sentence concise output.
Router Agent: Uses Lite fusing image description, text and transcript into a precise small search query. This is a critical design choice: it embeds a semantically dense query that retrieves more relevant passages.
RAG Engine: The rewritten query is embedded again using SigLip. Top 5 passages are retrieved from ChromaDB.
Diagnosis LLM: Pro receives the full context: retrieved passages, the user query, the image, and the transcribed audio. It is prompted to return strictly-typed JSON (summary, step-by-step diagnostics, page references, safety warnings), validated through Pydantic before the API Structured response is dispatched.

## 3. Tech Stack
PrusaAssist uses FastAPI for an asynchronous backend, routing multimodal inputs (text, audio, images) to Google Gemini agents (Flash/Lite/Pro) for specialized reasoning tasks. It implements a true multimodal RAG by utilizing PyMuPDF and LangChain to extract content, SigLIP (via PyTorch) to embed both text and diagrams into the same vector space, and ChromaDB for local, low-latency retrieval. The decoupled architecture ensures safe edge deployment by enforcing strict Pydantic JSON schemas to prevent frontend failure.

## 4. Key Design Decision & Tradeoffs
4.1) Choosing SigLIP over text-only embeddings: The deliberate choice of SigLIP - a vision-language model. This means a query about a visually identifiable problem can retrieve the diagram showing it, not just text describing it. The tradeoff is higher RAM at startup (model loaded once, cached) and a fixed 64-token text context window, mitigated by query rewriting to produce short, dense queries.
4.2) Tiered Model Strategy: Three Gemini models are used; Lite for simpler tasks (cheap, simple, fast), Flash for vision and audio processing (high quality perception, medium latency), Pro for synthesis (Best Reasoning, bad latency). This tiered approach avoids the common mistake of using one expensive model for every task, reducing both cost and latency without sacrificing output quality where it matters most.
4.3) Per-Model Collection Isolation: Rather a simple vector store, each Prusa Printer Model has its own ChromaDB collection. Meaning retrieval is scoped to correct manuals - the /health endpoint explicitly reports collection status and chunk counts, enabling rapid debugging of ingestion issues before any user query is served.
4.4) Query Rewriting: Embedding a raw user utterance gives poor retrieval. The Router Agent reduces this to a precise technical query before embedding. This simple design choice has the greatest impact on retrieval quality.
4.5) Graceful Degradation, Modular Architecture and logging: Each service stage wraps its API calls in a try/except block. If one service fails, the rest continues with the rest of the inputs. Debugging is simple now as each task has its own service and logging is done at each step, therefore constant updates about the systems keep coming.
4.6) Pydantic Validation: The schema enforces types at both input and output. A field_validator coerces unexpected LLM outputs (e.g., a string 'N/A' where a list is expected) into the correct Python types before they reach the API response. This prevents the system from crashing on malformed model outputs — a common failure mode in production LLM applications. The AnalyzeResponse schema similarly documents the exact contract that the frontend can rely on.

## 5) What Works:
End-to-end pipeline        Multimodal input       Cross-modal retrieval          Structured and grounded output Clean API contract         Observable pipeline    Clear Modular Architecture     Multiple agents used for queries
Retrieval / Chunking from real technical manuals   