"""Call the API from another program. It only needs `requests`.

Start the API first (python run.py, or uvicorn app.api:app), then:

    python examples/client_example.py "What is the minimum attendance?"
    python examples/client_example.py notice.png scan.pdf "When is the fee due?"

Files are added to the index before the question is asked. A file that was
indexed before is recognised by its content and is not read again.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/") + "/api/v1"


def add_files(paths: list[Path]) -> None:
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in paths]
    response = requests.post(f"{BASE}/index", files=files, timeout=1800)
    response.raise_for_status()
    body = response.json()
    print("added:", ", ".join(body["newly_indexed"]) or "nothing new")


def ask(question: str) -> dict:
    response = requests.post(f"{BASE}/ask", data={"question": question}, timeout=300)
    response.raise_for_status()
    return response.json()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    *file_args, question = sys.argv[1:]
    if file_args:
        add_files([Path(p) for p in file_args])

    result = ask(question)
    print(f"\n{result['answer']}\n")
    print(f"confidence: {result['confidence']}  (evidence strength {result['retrieval_score']:.2f})")
    if not result["evidence_sufficient"]:
        print(f"why: {result['reason']}")
    for number, cite in enumerate(result["citations"], start=1):
        print(f"[{number}] {cite['document_name']}, page {cite['page']}, {cite['section']}")


if __name__ == "__main__":
    main()
