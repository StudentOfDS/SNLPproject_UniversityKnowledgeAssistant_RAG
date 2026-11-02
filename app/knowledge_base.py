"""Indexing and retrieval: chunking, TF-IDF, query likelihood, sentence embeddings (FAISS), fusion and reranking."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import joblib
import numpy as np
from scipy import sparse
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import ingest
from app.evidence import Evidence, assess_evidence
from app.models import Citation
from app.ocr import get_engine
from app.settings import Settings
from app.lm_retrieval import QueryLikelihoodIndex
from app.snlp.classify import QuestionRouter
from app.text_utils import content_stems, set_stemmer, tfidf_analyzer
from app.understanding import QueryAnalysis, analyze

LOGGER = logging.getLogger(__name__)

MIN_SECTION_WORDS = 25
INDEX_VERSION = 4

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
}


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    source: str
    page: int
    section: str
    chunk_index: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # value used for ordering in the chosen mode
    retrieval_mode: str
    tfidf_score: float = 0.0
    semantic_score: float | None = None
    lm_score: float | None = None

    @property
    def relevance(self) -> float:
        """Raw similarity of the chunk to the question (not a rank-based number)."""
        return self.semantic_score if self.semantic_score is not None else self.tfidf_score


@dataclass
class SharedIndex:
    """Everything built across all searched documents at once, so scores are comparable between files."""
    vectorizer: TfidfVectorizer
    matrix: object
    lm: QueryLikelihoodIndex | None
    router: QuestionRouter | None


@dataclass
class IndexData:
    chunks: list[Chunk]
    vectorizer: TfidfVectorizer
    matrix: object
    faiss_index: object | None
    vocabulary: set[str] = field(default_factory=set)


class KnowledgeBase:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.storage_dir = self.settings.storage_path
        self.documents_dir = self.storage_dir / "documents"
        self.indexes_dir = self.storage_dir / "indexes"
        self.metadata_dir = self.storage_dir / "metadata"
        for directory in (self.documents_dir, self.indexes_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # Models load on first use so the API starts fast and still works offline (TF-IDF only).
        self._embedding_model: SentenceTransformer | None = None
        self._embedding_model_unavailable = False
        self._reranker: CrossEncoder | None = None
        self._reranker_unavailable = False
        self._cache: dict[str, IndexData] = {}
        self._shared: dict[tuple, SharedIndex] = {}
        self._lock = threading.RLock()

    # models -------------------------------------------------------------

    @property
    def embedding_model(self) -> SentenceTransformer | None:
        if not self.settings.semantic_search:
            return None
        if self._embedding_model is not None or self._embedding_model_unavailable:
            return self._embedding_model
        try:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        except Exception:
            LOGGER.warning(
                "Could not load embedding model '%s' (no internet on first run?). Using TF-IDF only.",
                self.settings.embedding_model,
            )
            self._embedding_model_unavailable = True
        return self._embedding_model

    @property
    def reranker(self) -> CrossEncoder | None:
        if not self.settings.use_reranker:
            return None
        if self._reranker is not None or self._reranker_unavailable:
            return self._reranker
        try:
            self._reranker = CrossEncoder(self.settings.reranker_model)
        except Exception:
            LOGGER.warning("Could not load reranker '%s', skipping reranking.", self.settings.reranker_model)
            self._reranker_unavailable = True
        return self._reranker

    # paths and bookkeeping ----------------------------------------------

    clean_text = staticmethod(ingest.clean_text)

    @staticmethod
    def file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def safe_filename(file_name: str | None) -> str:
        return Path(file_name or "uploaded.pdf").name or "uploaded.pdf"

    def metadata_path(self, file_hash: str) -> Path:
        return self.metadata_dir / f"{file_hash}.json"

    def index_path(self, file_hash: str) -> Path:
        return self.indexes_dir / file_hash

    def document_path(self, file_hash: str, file_name: str) -> Path:
        return self.documents_dir / f"{file_hash}_{self.safe_filename(file_name)}"

    @property
    def signature(self) -> str:
        s = self.settings
        return f"v{INDEX_VERSION}|{s.chunk_size}|{s.chunk_overlap}|{s.embedding_model}"

    def is_indexed(self, file_hash: str) -> bool:
        directory = self.index_path(file_hash)
        return (
            self.metadata_path(file_hash).exists()
            and (directory / "chunks.json").exists()
            and (directory / "tfidf_vectorizer.joblib").exists()
            and (directory / "tfidf_matrix.joblib").exists()
        )

    def has_semantic_index(self, file_hash: str) -> bool:
        return (self.index_path(file_hash) / "semantic.faiss").exists()

    def _needs_rebuild(self, file_hash: str) -> bool:
        metadata = self._load_metadata(file_hash)
        if metadata.get("index_signature") != self.signature:
            return True
        # pages skipped earlier for lack of OCR can be read now
        return bool(metadata.get("skipped_ocr_pages")) and get_engine(self.settings.ocr_engine) is not None

    # indexing -----------------------------------------------------------

    def process_and_index(self, file_name: str, file_bytes: bytes) -> dict:
        file_name = self.safe_filename(file_name)
        ingest.validate_upload(file_name, file_bytes, self.settings.max_file_size_mb)
        file_hash = self.file_hash(file_bytes)

        with self._lock:
            if self.is_indexed(file_hash) and not self._needs_rebuild(file_hash):
                self._add_missing_semantic_index(file_hash)
                metadata = self._load_metadata(file_hash)
                metadata["reused"] = True
                return metadata

            self.remove_index(file_hash)
            document_id = f"DOC-{file_hash[:12].upper()}"
            document_type = self.infer_document_type(file_name, file_bytes)
            document_path = self.document_path(file_hash, file_name)
            document_path.write_bytes(file_bytes)

            try:
                extraction = ingest.extract_pages(
                    file_name, file_bytes, self.settings, get_engine(self.settings.ocr_engine)
                )
                chunks = self.chunk_pages(extraction.pages, file_hash, document_id, file_name, document_type)
                if not chunks:
                    raise ValueError(f"No readable text was found in {file_name}.")

                directory = self.index_path(file_hash)
                directory.mkdir(parents=True, exist_ok=True)
                self._save_chunks(directory / "chunks.json", chunks)
                self._build_tfidf_index(file_hash, chunks)
                self._build_semantic_index(file_hash, chunks)

                metadata = {
                    "document_id": document_id,
                    "document_name": file_name,
                    "document_type": document_type,
                    "file_hash": file_hash,
                    "size_bytes": len(file_bytes),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": len(chunks),
                    "page_count": len(extraction.pages),
                    "ocr_pages": [p.number for p in extraction.pages if p.ocr_used],
                    "skipped_ocr_pages": extraction.skipped_ocr_pages,
                    "semantic_index": self.has_semantic_index(file_hash),
                    "index_signature": self.signature,
                    "reused": False,
                }
                self._save_metadata(file_hash, metadata)
                return metadata
            except Exception:
                document_path.unlink(missing_ok=True)
                self.remove_index(file_hash)
                raise

    def chunk_pages(self, pages, file_hash, document_id, document_name, document_type) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            index_on_page = 0
            for section, section_text in self._merge_tiny_sections(self.split_into_sections(page.text)):
                for piece in self.chunk_text(section_text):
                    chunks.append(Chunk(
                        chunk_id=f"{file_hash[:12]}-P{page.number}-C{index_on_page:03d}",
                        document_id=document_id, document_name=document_name,
                        document_type=document_type, source=document_name,
                        page=page.number, section=section, chunk_index=index_on_page, text=piece,
                    ))
                    index_on_page += 1
        return chunks

    @staticmethod
    def _merge_tiny_sections(sections: list[tuple[str, str]], min_words: int = MIN_SECTION_WORDS) -> list[tuple[str, str]]:
        """Very short sections (a heading with one line under it) go into the section before them.

        As separate chunks they score too well in keyword search, because cosine similarity favours short texts.
        """
        merged: list[tuple[str, str]] = []
        for title, body in sections:
            if merged and len(body.split()) < min_words:
                previous_title, previous_body = merged[-1]
                extra = body if title == "General" else f"{title}\n{body}"
                merged[-1] = (previous_title, f"{previous_body}\n{extra}")
            else:
                merged.append((title, body))
        return merged

    @staticmethod
    def looks_like_heading(line: str) -> bool:
        line = line.strip()
        if not line or len(line) > 120 or line.endswith((",", ";")):
            return False

        keywords = ("section", "chapter", "unit", "part", "article", "rule", "regulation",
                    "policy", "procedure", "schedule", "appendix")
        if re.match(rf"^({'|'.join(keywords)})\b", line, re.IGNORECASE):
            return True

        numbered = re.match(r"^(\d+)(\.\d+)*[.)]?\s+([A-Z].*)", line)
        if numbered:
            words = line.split()
            # "12 July 2026" and similar date lines are not headings
            if int(numbered.group(1)) <= 50 and not any(w.lower().strip(",.") in _MONTHS for w in words[:3]):
                return len(words) <= 14
            return False

        words = line.split()
        return 1 <= len(words) <= 8 and line.upper() == line and any(c.isalpha() for c in line) \
            and not any(c.isdigit() for c in line)

    @classmethod
    def split_into_sections(cls, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        title = "General"
        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if cls.looks_like_heading(line):
                if lines:
                    sections.append((title, "\n".join(lines)))
                    lines = []
                title = line
            elif line:
                lines.append(line)
        if lines:
            sections.append((title, "\n".join(lines)))
        return [(t, body) for t, body in sections if body.strip()] or [("General", text)]

    def chunk_text(self, text: str) -> list[str]:
        """Pack whole lines into chunks of about chunk_size words, repeating the last
        `chunk_overlap` words worth of lines at the start of the next chunk."""
        size = self.settings.chunk_size
        overlap = min(max(0, self.settings.chunk_overlap), size - 1)

        units: list[tuple[str, int]] = []
        for line in text.splitlines():
            words = line.split()
            if not words:
                continue
            if len(words) <= size:
                units.append((" ".join(words), len(words)))
                continue
            for start in range(0, len(words), size - overlap):  # one very long line
                piece = words[start:start + size]
                units.append((" ".join(piece), len(piece)))
                if start + size >= len(words):
                    break

        chunks: list[str] = []
        current: list[tuple[str, int]] = []
        count = 0
        for line, n in units:
            if current and count + n > size:
                chunks.append("\n".join(item[0] for item in current))
                kept: list[tuple[str, int]] = []
                kept_words = 0
                for item in reversed(current):
                    if kept_words + item[1] > overlap:
                        break
                    kept.insert(0, item)
                    kept_words += item[1]
                current, count = kept, kept_words
            current.append((line, n))
            count += n
        if current:
            chunks.append("\n".join(item[0] for item in current))
        return chunks

    @staticmethod
    def infer_document_type(file_name: str, file_bytes: bytes) -> str:
        patterns = {
            "academic_regulation": ["regulation", "academic rule"],
            "examination_rule": ["exam", "examination"],
            "hostel_rule": ["hostel", "residence", "residential"],
            "scholarship": ["scholarship", "financial aid"],
            "academic_calendar": ["calendar", "academic year", "holiday"],
            "circular": ["circular", "notice", "notification"],
            "course_handbook": ["course handbook", "course_catalogue", "curriculum"],
            "student_handbook": ["resource book", "srb", "handbook", "student guide"],
            "policy": ["policy", "policies"],
        }
        name = file_name.lower()
        for document_type, words in patterns.items():
            if any(w in name for w in words):
                return document_type

        if ingest.extension_of(file_name) == ".pdf":
            try:
                import pymupdf
                with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
                    preview = " ".join((page.get_text("text") or "") for page in list(pdf)[:2]).lower()
                for document_type, words in patterns.items():
                    if any(w in preview for w in words):
                        return document_type
            except Exception:
                LOGGER.debug("Could not read a preview of %s to guess its type.", file_name)
        return "unclassified"

    def _build_tfidf_index(self, file_hash: str, chunks: list[Chunk]) -> None:
        vectorizer = TfidfVectorizer(analyzer=tfidf_analyzer, sublinear_tf=True, max_features=100_000)
        matrix = vectorizer.fit_transform([f"{c.section} {c.text}" for c in chunks])
        directory = self.index_path(file_hash)
        joblib.dump(vectorizer, directory / "tfidf_vectorizer.joblib")
        joblib.dump(matrix, directory / "tfidf_matrix.joblib")

    def _build_semantic_index(self, file_hash: str, chunks: list[Chunk]) -> None:
        model = self.embedding_model
        if model is None:
            return
        embeddings = model.encode(
            [f"{c.section}. {c.text}" for c in chunks],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=32,
        ).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path(file_hash) / "semantic.faiss"))

    def _add_missing_semantic_index(self, file_hash: str) -> None:
        if self.has_semantic_index(file_hash) or self.embedding_model is None:
            return
        self._build_semantic_index(file_hash, self._load_chunks(self.index_path(file_hash) / "chunks.json"))
        self._cache.pop(file_hash, None)
        self._shared.clear()
        metadata = self._load_metadata(file_hash)
        metadata["semantic_index"] = self.has_semantic_index(file_hash)
        self._save_metadata(file_hash, metadata)

    def _load_index(self, file_hash: str) -> IndexData:
        with self._lock:
            cached = self._cache.get(file_hash)
            if cached is not None:
                return cached
            directory = self.index_path(file_hash)
            chunks = self._load_chunks(directory / "chunks.json")
            faiss_file = directory / "semantic.faiss"
            vocabulary: set[str] = set()
            for chunk in chunks:
                vocabulary.update(content_stems(f"{chunk.document_name} {chunk.section} {chunk.text}", frozenset()))
            data = IndexData(
                chunks=chunks,
                vectorizer=joblib.load(directory / "tfidf_vectorizer.joblib"),
                matrix=joblib.load(directory / "tfidf_matrix.joblib"),
                faiss_index=faiss.read_index(str(faiss_file)) if faiss_file.exists() else None,
                vocabulary=vocabulary,
            )
            self._cache[file_hash] = data
            return data

    # retrieval ----------------------------------------------------------

    def vocabulary(self, file_hashes: list[str]) -> set[str]:
        words: set[str] = set()
        for file_hash in file_hashes:
            if self.is_indexed(file_hash):
                words |= self._load_index(file_hash).vocabulary
        return words

    def search(self, question: str, file_hashes: list[str], mode: str = "hybrid",
               top_k: int | None = None, analysis: QueryAnalysis | None = None) -> list[RetrievedChunk]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if mode not in {"tfidf", "semantic", "lm", "hybrid"}:
            raise ValueError("Retrieval mode must be tfidf, semantic, lm or hybrid.")

        top_k = top_k or self.settings.final_top_k
        hashes = [h for h in file_hashes if self.is_indexed(h)]
        if not hashes:
            return []
        set_stemmer(self.settings.stemmer)
        analysis = analysis or analyze(question)
        if not analysis.terms:
            return []

        chunks: list[Chunk] = []
        semantic_parts: list[np.ndarray] = []
        query_vector = None
        if mode in {"semantic", "hybrid"} and any(self.has_semantic_index(h) for h in hashes) \
                and self.embedding_model is not None:
            query_vector = self.embedding_model.encode(
                [question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")

        for file_hash in hashes:
            data = self._load_index(file_hash)
            n = len(data.chunks)
            chunks.extend(data.chunks)
            part = np.full(n, np.nan, dtype="float32")
            if query_vector is not None and data.faiss_index is not None:
                scores, ids = data.faiss_index.search(query_vector, n)
                for score, position in zip(scores[0], ids[0]):
                    if position >= 0:
                        part[position] = score
            semantic_parts.append(part)

        shared = self._shared_index(hashes, chunks)
        tfidf = self._tfidf_scores(shared, analysis)
        semantic = np.concatenate(semantic_parts)
        has_semantic = not np.all(np.isnan(semantic))

        lm = None
        if shared.lm is not None and mode in {"lm", "hybrid"}:
            lm = shared.lm.scores(analysis.terms)
            lm[~shared.lm.matches(analysis.terms)] = -np.inf

        def build(position: int, score: float, label: str) -> RetrievedChunk:
            sem = None if np.isnan(semantic[position]) else float(semantic[position])
            lm_value = None if lm is None or not np.isfinite(lm[position]) else float(lm[position])
            return RetrievedChunk(chunks[position], score, label, float(tfidf[position]), sem, lm_value)

        tfidf_order = [int(i) for i in np.argsort(-tfidf)[: self.settings.tfidf_top_k] if tfidf[i] > 0]
        semantic_order = []
        if has_semantic:
            filled = np.nan_to_num(semantic, nan=-1.0)
            semantic_order = [int(i) for i in np.argsort(-filled)[: self.settings.semantic_top_k] if filled[i] > 0]
        lm_order = []
        if lm is not None:
            lm_order = [int(i) for i in np.argsort(-lm)[: self.settings.tfidf_top_k] if np.isfinite(lm[i])]

        if mode == "tfidf" or (mode == "semantic" and not has_semantic):
            results = [build(i, float(tfidf[i]), "tfidf") for i in tfidf_order]
        elif mode == "semantic":
            results = [build(i, float(semantic[i]), "semantic") for i in semantic_order]
        elif mode == "lm":
            results = [build(i, float(lm[i]), "lm") for i in lm_order]
        else:
            posterior = self._document_posterior(shared, question) if self.settings.use_router else {}
            results = self._fuse([semantic_order, tfidf_order, lm_order], chunks, posterior, build)

        if self.reranker and results:
            results = self._rerank(question, results)
        return results[:top_k]

    def _tfidf_scores(self, shared: SharedIndex, analysis: QueryAnalysis) -> np.ndarray:
        """Cosine similarity between the passages and the weighted question vector."""
        vocabulary = shared.vectorizer.vocabulary_
        idf = shared.vectorizer.idf_
        weights: dict[str, float] = dict(analysis.terms)
        for pair in analysis.pairs:
            weights[pair] = 1.0

        columns, values = [], []
        for term, weight in weights.items():
            column = vocabulary.get(term)
            if column is not None:
                columns.append(column)
                values.append(weight * idf[column])
        if not columns:
            return np.zeros(shared.matrix.shape[0])
        vector = sparse.csr_matrix((values, ([0] * len(columns), columns)), shape=(1, len(vocabulary)))
        norm = np.sqrt(vector.multiply(vector).sum())
        return np.asarray((shared.matrix @ vector.T).todense()).ravel() / norm

    def likely_documents(self, question: str, file_hashes: list[str]) -> list[dict]:
        """Which documents the Naive Bayes router thinks the question belongs to."""
        hashes = [h for h in file_hashes if self.is_indexed(h)]
        if not hashes or not self.settings.use_router:
            return []
        chunks = [c for h in hashes for c in self._load_index(h).chunks]
        router = self._shared_index(hashes, chunks).router
        return router.route(question, top=3) if router else []

    def _document_posterior(self, shared: SharedIndex, question: str) -> dict[str, float]:
        if shared.router is None:
            return {}
        return {r["document"]: r["probability"] for r in shared.router.route(question, top=10)}

    def _shared_index(self, hashes: list[str], chunks: list[Chunk]) -> SharedIndex:
        """One keyword space over all searched documents.

        Scores from separate per-file indexes can't be compared: a word that is rare in a
        long document looks common in a one-page file and the other way round.
        """
        s = self.settings
        key = (tuple(hashes), s.stemmer, s.lm_smoothing, s.lm_mu, s.use_lm_retriever, s.use_router)
        cached = self._shared.get(key)
        if cached is None:
            texts = [f"{c.document_name} {c.section} {c.text}" for c in chunks]
            vectorizer = TfidfVectorizer(analyzer=tfidf_analyzer, sublinear_tf=True, max_features=100_000)
            matrix = vectorizer.fit_transform(texts)
            lm = QueryLikelihoodIndex(texts, s.lm_smoothing, s.lm_mu) if s.use_lm_retriever else None
            router = None
            if s.use_router and len({c.document_name for c in chunks}) >= 2:
                router = QuestionRouter().fit(chunks)
            if len(self._shared) >= 8:
                self._shared.clear()
            cached = self._shared[key] = SharedIndex(vectorizer, matrix, lm, router)
        return cached

    def _fuse(self, rankings: list[list[int]], chunks: list[Chunk], posterior: dict[str, float], build) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion over the rankings that exist, then a nudge toward the likely document."""
        rankings = [r for r in rankings if r]
        if not rankings:
            return []
        k = self.settings.rrf_k
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, position in enumerate(ranking, start=1):
                fused[position] = fused.get(position, 0.0) + 1.0 / (k + rank)
        best_possible = len(rankings) / (k + 1)
        for position in fused:
            share = posterior.get(chunks[position].document_name, 0.0)
            fused[position] *= 1.0 + self.settings.router_boost * share
        ordered = sorted(fused, key=fused.get, reverse=True)[: self.settings.hybrid_top_k]
        return [build(i, fused[i] / best_possible, "hybrid") for i in ordered]

    def _rerank(self, question: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raw = self.reranker.predict([[question, r.chunk.text] for r in results], show_progress_bar=False)
        for result, value in zip(results, raw):
            result.score = float(1.0 / (1.0 + np.exp(-value)))
            result.retrieval_mode = "reranked"
        return sorted(results, key=lambda r: r.score, reverse=True)

    # answers ------------------------------------------------------------

    def assess(self, question: str, retrieved: list[RetrievedChunk], file_hashes: list[str],
               llm_available: bool = False) -> Evidence:
        return assess_evidence(question, retrieved, self.vocabulary(file_hashes), self.settings, llm_available)

    @staticmethod
    def to_citation(result: RetrievedChunk) -> Citation:
        chunk = result.chunk
        return Citation(
            chunk_id=chunk.chunk_id, document_id=chunk.document_id, document_name=chunk.document_name,
            document_type=chunk.document_type, section=chunk.section, page=chunk.page,
            score=round(result.relevance, 4), text=chunk.text,
        )

    # storage ------------------------------------------------------------

    @staticmethod
    def _save_chunks(path: Path, chunks: list[Chunk]) -> None:
        path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def _load_chunks(path: Path) -> list[Chunk]:
        return [Chunk(**item) for item in json.loads(path.read_text(encoding="utf-8"))]

    def _save_metadata(self, file_hash: str, metadata: dict) -> None:
        self.metadata_path(file_hash).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_metadata(self, file_hash: str) -> dict:
        return json.loads(self.metadata_path(file_hash).read_text(encoding="utf-8"))

    def remove_index(self, file_hash: str) -> None:
        with self._lock:
            self._cache.pop(file_hash, None)
            self._shared.clear()
            directory = self.index_path(file_hash)
            if directory.exists():
                shutil.rmtree(directory)
            self.metadata_path(file_hash).unlink(missing_ok=True)

    def list_documents(self) -> list[dict]:
        documents = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            try:
                documents.append(self._load_metadata(path.stem))
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("Could not read metadata file %s", path)
        return documents

    def all_chunks(self, file_hashes: list[str] | None = None) -> list[Chunk]:
        hashes = file_hashes if file_hashes is not None else [d["file_hash"] for d in self.list_documents()]
        chunks: list[Chunk] = []
        for file_hash in hashes:
            if self.is_indexed(file_hash):
                chunks.extend(self._load_index(file_hash).chunks)
        return chunks
