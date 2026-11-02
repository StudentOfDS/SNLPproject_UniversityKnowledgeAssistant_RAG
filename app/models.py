from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    section: str
    page: int
    score: float  # similarity of this passage to the question
    text: str
    highlights: list[str] = Field(default_factory=list)  # the sentences the answer is based on


class AnswerResponse(BaseModel):
    query: str
    answer: str
    retrieval_score: float  # overall evidence strength, 0 to 1
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    evidence_sufficient: bool  # False means the answer is a "not in your documents" reply
    reason: str = ""
    signals: dict = Field(default_factory=dict)
    analysis: dict = Field(default_factory=dict)  # how the question was read
    likely_documents: list[dict] = Field(default_factory=list)  # Naive Bayes guess of where it belongs
    suggestions: list[str] = Field(default_factory=list)  # topics the files do cover (only when refused)
    citations: list[Citation] = Field(default_factory=list)
    processed_documents: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    section: str
    page: int
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult] = Field(default_factory=list)


class IndexedDocument(BaseModel):
    document_id: str
    document_name: str
    document_type: str
    file_hash: str
    size_bytes: int
    indexed_at: str
    chunk_count: int
    page_count: int = 0
    ocr_pages: list[int] = Field(default_factory=list)


class IndexResponse(BaseModel):
    documents: list[IndexedDocument] = Field(default_factory=list)
    newly_indexed: list[str] = Field(default_factory=list)
    reused: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int
    llm_configured: bool
    semantic_search: bool
    ocr_engine: str | None
