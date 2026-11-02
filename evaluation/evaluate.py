"""Measure the pipeline on the real files in documents/.

    python -m evaluation.evaluate            # report
    python -m evaluation.evaluate --save     # also write evaluation/latest_results.json

Relevance is judged by phrases that must appear in a retrieved chunk (see benchmark.json), so the
benchmark keeps working when chunk sizes change. The numbers are a sanity check on a small
benchmark, not a general accuracy figure.

The report has three parts:
  1. retrieval, with the components switched on one at a time (what each course technique adds)
  2. stemmers and smoothing methods compared on the same benchmark
  3. the whole pipeline: answered, refused, wrongly answered, sentence and Yes/No quality
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import answering, understanding  # noqa: E402
from app.generator import GroundedAnswerGenerator  # noqa: E402
from app.ingest import SUPPORTED_EXTENSIONS  # noqa: E402
from app.knowledge_base import KnowledgeBase  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.snlp import data  # noqa: E402

TOP_K = 5


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def is_relevant(chunk_text: str, phrases: list[str]) -> bool:
    text = squash(chunk_text)
    return any(squash(p) in text for p in phrases)


def build(settings: Settings) -> tuple[KnowledgeBase, list[str]]:
    kb = KnowledgeBase(settings)
    for path in sorted(settings.documents_path.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            kb.process_and_index(path.name, path.read_bytes())
    return kb, [d["file_hash"] for d in kb.list_documents()]


def retrieval(kb, hashes, items, mode: str, **changes) -> dict:
    original = kb.settings
    kb.settings = original.model_copy(update=changes)
    hits, reciprocal = 0, 0.0
    for item in items:
        found = kb.search(item["question"], hashes, mode=mode, top_k=TOP_K)
        rank = next((i for i, r in enumerate(found, 1) if is_relevant(r.chunk.text, item["relevant_phrases"])), None)
        if rank:
            hits += 1
            reciprocal += 1 / rank
    kb.settings = original
    n = len(items)
    return {f"recall@{TOP_K}": round(hits / n, 3), "mrr": round(reciprocal / n, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    benchmark = json.loads((Path(__file__).parent / "benchmark.json").read_text(encoding="utf-8"))
    answerable, unanswerable = benchmark["answerable"], benchmark["unanswerable"]

    settings = Settings()
    data.configure(settings.nltk_data_path)
    kb, hashes = build(settings)
    generator = GroundedAnswerGenerator(settings)
    llm = generator.is_configured

    corpus = squash(" ".join(c.text for c in kb.all_chunks()))
    broken = [i["question"] for i in answerable if not any(squash(p) in corpus for p in i["relevant_phrases"])]
    if broken:
        print("These benchmark questions have no matching phrase in the documents (fix benchmark.json):")
        for question in broken:
            print("  -", question)

    report: dict = {
        "documents": [d["document_name"] for d in kb.list_documents()],
        "chunks": len(kb.all_chunks()),
        "semantic_search": kb.embedding_model is not None,
        "llm": settings.llm_model if llm else None,
    }

    # 1. components, switched on one at a time
    report["retrieval"] = {
        "tf-idf only": retrieval(kb, hashes, answerable, "tfidf"),
        "language model only": retrieval(kb, hashes, answerable, "lm"),
        "tf-idf + language model (rank fusion)": retrieval(kb, hashes, answerable, "hybrid", use_router=False),
        "+ Naive Bayes document router": retrieval(kb, hashes, answerable, "hybrid", use_router=True),
    }
    if kb.embedding_model is not None:
        report["retrieval"]["semantic only"] = retrieval(kb, hashes, answerable, "semantic")

    # 2. choices that the course topics ask you to compare
    report["stemmers"] = {name: retrieval(kb, hashes, answerable, "tfidf", stemmer=name)
                          for name in ("snowball", "porter", "lancaster", "none")}
    report["lm_smoothing"] = {name: retrieval(kb, hashes, answerable, "lm", lm_smoothing=name)
                              for name in ("dirichlet", "jelinek_mercer", "laplace")}

    # 3. the pipeline as a user meets it
    answered = keyword_hits = cited_ok = 0
    wrongly_refused: list[str] = []
    verdicts = {"correct": 0, "wrong": 0, "none": 0}
    for item in answerable:
        analysis = understanding.analyze(item["question"])
        found = kb.search(item["question"], hashes, mode="hybrid", top_k=TOP_K, analysis=analysis)
        verdict = kb.assess(item["question"], found, hashes, llm_available=llm)
        citations = [kb.to_citation(r) for r in found]
        answer = generator.generate(item["question"], citations, analysis) if verdict.sufficient else None
        if answer is None:
            wrongly_refused.append(item["question"])
            continue
        answered += 1
        keyword_hits += any(k.lower() in answer.lower() for k in item["answer_keywords"])
        cited_ok += any(is_relevant(c.text, item["relevant_phrases"]) for c in citations)
        if "expected_polarity" in item:
            given = answering.polarity(analysis, answering.choose_sentences(analysis, citations))
            key = "none" if not given else "correct" if given.lower() == item["expected_polarity"] else "wrong"
            verdicts[key] += 1

    refused, wrongly_answered = 0, []
    for question in unanswerable:
        analysis = understanding.analyze(question)
        found = kb.search(question, hashes, mode="hybrid", top_k=TOP_K, analysis=analysis)
        verdict = kb.assess(question, found, hashes, llm_available=llm)
        answer = None
        if verdict.sufficient:
            answer = generator.generate(question, [kb.to_citation(r) for r in found], analysis)
        if answer is None:
            refused += 1
        else:
            wrongly_answered.append(question)

    report["pipeline"] = {
        "answerable_questions": len(answerable),
        "answered": answered,
        "answers_with_expected_keyword": keyword_hits,
        "answers_citing_a_relevant_chunk": cited_ok,
        "wrongly_refused": wrongly_refused,
        "unanswerable_questions": len(unanswerable),
        "refused": refused,
        "wrongly_answered": wrongly_answered,
        "yes_no_verdicts_among_answered": verdicts,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.save:
        out = Path(__file__).parent / "latest_results.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
