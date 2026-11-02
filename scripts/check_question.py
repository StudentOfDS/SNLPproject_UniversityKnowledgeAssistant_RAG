"""Show how the "is this in the documents" check sees a question.

    python scripts/check_question.py "What is the attendance rule?" "Who won the world cup?"

It indexes documents/ (cached after the first run), searches, and prints the numbers
the check uses. Use it to tune the thresholds in .env for your own documents.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import SUPPORTED_EXTENSIONS  # noqa: E402
from app.knowledge_base import KnowledgeBase  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.understanding import analyze  # noqa: E402


def main(questions: list[str]) -> None:
    settings = Settings()
    kb = KnowledgeBase(settings)
    for path in sorted(settings.documents_path.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            kb.process_and_index(path.name, path.read_bytes())
    hashes = [d["file_hash"] for d in kb.list_documents()]
    llm = bool(settings.google_api_key.strip())

    for question in questions:
        analysis = analyze(question)
        found = kb.search(question, hashes, mode="hybrid", top_k=settings.final_top_k, analysis=analysis)
        verdict = kb.assess(question, found, hashes, llm_available=llm)
        print(f"\n{question}")
        print(f"  answer it:  {'yes' if verdict.sufficient else 'no'} ({verdict.confidence}, strength {verdict.strength:.2f})")
        print(f"  reason:     {verdict.reason}")
        print(f"  question:   {analysis.question_type}, terms {sorted(analysis.terms)}")
        print("  router:     " + ", ".join(f"{x['document']} {x['probability']:.0%}" for x in kb.likely_documents(question, hashes)[:2]))
        print(f"  known terms {verdict.known_term_ratio:.2f} | best coverage {verdict.term_coverage:.2f} | "
              f"semantic {verdict.semantic_similarity} | tf-idf {verdict.tfidf_similarity:.2f}")
        if found:
            top = found[0].chunk
            print(f"  best chunk: {top.document_name}, page {top.page}, {top.section}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
