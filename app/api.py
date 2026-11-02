from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.evidence import NOT_FOUND_MESSAGE, Evidence
from app.generator import GroundedAnswerGenerator
from app.ingest import SUPPORTED_EXTENSIONS
from app.knowledge_base import KnowledgeBase
from app.models import AnswerResponse, HealthResponse, IndexResponse, SearchResponse, SearchResult
from app import answering, understanding
from app.corpus import CorpusInsights
from app.ocr import get_engine
from app.settings import Settings
from app.snlp import data as nltk_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)

settings = Settings()
knowledge_base = KnowledgeBase(settings)
answer_generator = GroundedAnswerGenerator(settings)
insights = CorpusInsights(knowledge_base)
nltk_data.configure(settings.nltk_data_path)


def auto_index_folder(folder: Path) -> None:
    """Index every supported file directly inside `folder`. Files already indexed are skipped."""
    if not folder.exists():
        return
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        return

    LOGGER.info("Indexing %d file(s) from %s", len(paths), folder)
    for path in paths:
        try:
            metadata = knowledge_base.process_and_index(path.name, path.read_bytes())
            LOGGER.info("  %-15s %s", "already indexed" if metadata.get("reused") else "indexed", path.name)
        except ValueError as exc:
            LOGGER.warning("  skipped         %s (%s)", path.name, exc)
        except Exception:
            LOGGER.exception("  could not index %s", path.name)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.auto_index_on_startup:
        auto_index_folder(settings.documents_path)
        LOGGER.info("Ready with %d document(s).", len(knowledge_base.list_documents()))
    yield


app = FastAPI(
    title="University Knowledge Assistant API",
    description="Answers questions from university documents and always points to the source.",
    version="2.0.0",
    lifespan=lifespan,
)

# Other groups call this service from their own pages, so any origin is allowed by default.
_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins in {"", "*"} else [o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server. Check the server log."})


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = get_engine(settings.ocr_engine)
    return HealthResponse(
        status="ok",
        indexed_documents=len(knowledge_base.list_documents()),
        llm_configured=answer_generator.is_configured,
        semantic_search=knowledge_base.embedding_model is not None,
        ocr_engine=engine.name if engine else None,
    )


@app.get("/api/v1/documents")
def documents() -> list[dict]:
    return knowledge_base.list_documents()


@app.get("/api/v1/corpus")
def corpus() -> dict:
    """Collection statistics, keywords and a summary per document, and the topics found in the files."""
    if not knowledge_base.list_documents():
        raise HTTPException(status_code=400, detail="No documents are indexed yet.")
    return insights.profile()


@app.post("/api/v1/reindex")
def reindex() -> dict:
    """Pick up files added to documents/ without restarting."""
    before = len(knowledge_base.list_documents())
    auto_index_folder(settings.documents_path)
    after = len(knowledge_base.list_documents())
    return {"indexed_documents": after, "newly_indexed": max(0, after - before)}


@app.post("/api/v1/index", response_model=IndexResponse)
async def index_documents(
    files: Annotated[list[UploadFile], File(description="PDF or image files")],
) -> IndexResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Attach at least one file.")

    indexed, new, reused = [], [], []
    for upload in files:
        metadata = await _index_upload(upload)
        indexed.append(metadata)
        (reused if metadata.get("reused") else new).append(metadata["document_name"])
    return IndexResponse(documents=indexed, newly_indexed=new, reused=reused)


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(
    question: Annotated[str, Form(...)],
    mode: Annotated[str, Form()] = "hybrid",
    top_k: Annotated[int, Form()] = 5,
    files: Annotated[Optional[list[UploadFile]], File(description="Optional extra files to index first")] = None,
) -> SearchResponse:
    _validate(question, mode, top_k)
    hashes = await _resolve_hashes(files)
    results = knowledge_base.search(question, hashes, mode=mode, top_k=top_k)
    return SearchResponse(
        query=question,
        mode=mode,
        results=[
            SearchResult(
                chunk_id=r.chunk.chunk_id, document_id=r.chunk.document_id, document_name=r.chunk.document_name,
                document_type=r.chunk.document_type, section=r.chunk.section, page=r.chunk.page,
                score=round(r.relevance, 4), text=r.chunk.text,
            )
            for r in results
        ],
    )


@app.post("/api/v1/ask", response_model=AnswerResponse)
async def ask_question(
    question: Annotated[str, Form(...)],
    mode: Annotated[str, Form()] = "hybrid",
    history: Annotated[str, Form(description="JSON list of earlier {question, answer} turns")] = "[]",
    files: Annotated[Optional[list[UploadFile]], File(description="Optional extra files to index first")] = None,
) -> AnswerResponse:
    _validate(question, mode, settings.final_top_k)
    hashes = await _resolve_hashes(files)
    turns = _parse_history(history)

    if not hashes:
        raise HTTPException(
            status_code=400,
            detail="No documents are indexed yet. Put files in documents/ and call /api/v1/reindex, or attach them.",
        )

    # A short follow-up ("what about hostel residents?") needs the previous question to be searchable.
    # Longer questions stand on their own, so an off-topic one can't borrow words from the last turn.
    query = question
    if turns and len(understanding.analyze(question).terms) < 3:
        query = f"{turns[-1].get('question', '')} {question}".strip()

    analysis = understanding.analyze(query)
    retrieved = knowledge_base.search(query, hashes, mode=mode, top_k=settings.final_top_k, analysis=analysis)
    evidence = knowledge_base.assess(query, retrieved, hashes, llm_available=answer_generator.is_configured)
    names = [d["document_name"] for d in knowledge_base.list_documents() if d["file_hash"] in hashes]
    likely = knowledge_base.likely_documents(query, hashes)

    if not evidence.sufficient:
        return _not_found(question, evidence, names, analysis, likely)

    citations = [knowledge_base.to_citation(r) for r in retrieved]
    answer = answer_generator.generate(question, citations, analysis, turns)
    if answer is None:
        evidence.sufficient = False
        evidence.confidence = "LOW"
        evidence.reason = "The retrieved passages do not contain an answer to this question."
        return _not_found(question, evidence, names, analysis, likely)

    for chosen in answering.choose_sentences(analysis, citations):
        citations[chosen.citation_index].highlights.append(chosen.sentence)

    return AnswerResponse(
        query=question, answer=answer, retrieval_score=evidence.strength, confidence=evidence.confidence,
        evidence_sufficient=True, reason=evidence.reason, signals=evidence.as_dict(),
        analysis=analysis.as_dict(), likely_documents=likely,
        citations=citations, processed_documents=names,
    )


def _not_found(question: str, evidence: Evidence, names: list[str], analysis, likely: list[dict]) -> AnswerResponse:
    topic_model = insights.topics()
    suggestions = [t["label"] for t in topic_model.closest(question)] if topic_model else []
    return AnswerResponse(
        query=question, answer=NOT_FOUND_MESSAGE, retrieval_score=evidence.strength, confidence="LOW",
        evidence_sufficient=False, reason=evidence.reason, signals=evidence.as_dict(),
        analysis=analysis.as_dict(), likely_documents=likely, suggestions=suggestions,
        citations=[], processed_documents=names,
    )


def _parse_history(raw: str) -> list[dict]:
    try:
        parsed = json.loads(raw) if raw and raw.strip() else []
    except json.JSONDecodeError:
        return []
    return [t for t in parsed if isinstance(t, dict)] if isinstance(parsed, list) else []


def _validate(question: str, mode: str, top_k: int) -> None:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if mode not in {"tfidf", "semantic", "lm", "hybrid"}:
        raise HTTPException(status_code=400, detail="mode must be hybrid, tfidf, lm or semantic.")
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")


async def _index_upload(upload: UploadFile) -> dict:
    try:
        data = await upload.read()
        return knowledge_base.process_and_index(upload.filename or "upload.pdf", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Failed to index %s", upload.filename)
        raise HTTPException(status_code=500, detail=f"Could not process {upload.filename}.") from exc
    finally:
        await upload.close()


async def _resolve_hashes(files: Optional[list[UploadFile]]) -> list[str]:
    for upload in files or []:
        await _index_upload(upload)
    return [d["file_hash"] for d in knowledge_base.list_documents()]
