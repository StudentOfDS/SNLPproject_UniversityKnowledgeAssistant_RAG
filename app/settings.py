from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration lives here and is read from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (optional). Without a key the app quotes the best passage instead.
    google_api_key: str = ""
    llm_model: str = "gemini-flash-latest"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 30.0
    # The Gemini SDK retries on its own (5 attempts by default) which can turn a
    # flaky connection into a multi-minute wait, so we cap it.
    llm_max_attempts: int = 2

    # Retrieval
    # set to false to use keyword (TF-IDF) search only, e.g. when offline
    semantic_search: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    semantic_top_k: int = 10
    tfidf_top_k: int = 10
    hybrid_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60
    stemmer: str = "snowball"  # snowball | porter | lancaster | none

    # Third ranking: query likelihood with a smoothed unigram language model
    use_lm_retriever: bool = True
    lm_smoothing: str = "dirichlet"  # dirichlet | jelinek_mercer | laplace
    lm_mu: float = 120.0
    # Naive Bayes guess of the right document nudges the ranking
    use_router: bool = True
    router_boost: float = 0.5

    # Chunking. MiniLM reads about 256 word pieces, so chunks stay near 170 words.
    chunk_size: int = Field(default=170, ge=20)
    chunk_overlap: int = Field(default=30, ge=0)

    # Answer gate: when does the system say "not in your documents"
    # share of question terms that occur anywhere in the indexed documents
    min_known_term_ratio: float = 0.5
    # share of question terms found in the best matching chunk
    min_term_coverage: float = 0.5
    # a semantic match this strong is accepted even with low word overlap (paraphrases)
    semantic_strong: float = 0.60
    # below this cosine similarity a chunk is never treated as relevant
    semantic_floor: float = 0.20
    # minimum overall evidence strength. Higher without an LLM because nothing else
    # checks that the quoted passage really answers the question.
    min_strength_with_llm: float = 0.35
    min_strength_without_llm: float = 0.50

    # OCR
    ocr_engine: str = "auto"  # auto | rapidocr | tesseract | off
    ocr_dpi: int = 200
    # a page with less text than this is treated as a scan and read with OCR
    ocr_min_page_chars: int = 80
    # embedded pictures covering at least this share of the page are read too
    ocr_image_area_ratio: float = 0.1
    ocr_embedded_images: bool = True

    max_file_size_mb: int = 50
    storage_dir: str = "storage"
    documents_dir: str = "documents"
    auto_index_on_startup: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)

    @property
    def documents_path(self) -> Path:
        return Path(self.documents_dir)

    @property
    def nltk_data_path(self) -> Path:
        return self.storage_path / "nltk_data"
