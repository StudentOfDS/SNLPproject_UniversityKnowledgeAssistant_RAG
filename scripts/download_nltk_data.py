"""Download the NLTK sentence model (punkt) into storage/nltk_data.

    python scripts/download_nltk_data.py

Without it the app falls back to a simpler regex sentence splitter, so this is optional.
Run it once if you want the better splitter offline, or when building a Docker image.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import Settings  # noqa: E402
from app.snlp import data  # noqa: E402


def main() -> int:
    data.configure(Settings().nltk_data_path)
    failed = []
    for package, resource in data.REQUIRED.items():
        try:
            data.ensure(resource, package)
            print(f"ok       {package}")
        except data.DataUnavailable as exc:
            failed.append(package)
            print(f"missing  {package}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
