"""Finds or downloads the NLTK corpora the sentence splitter needs."""
from __future__ import annotations

import logging
from pathlib import Path

import nltk

LOGGER = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """Raised when a corpus is missing and could not be downloaded."""


_checked: dict[str, bool] = {}
_download_dir: Path | None = None


def configure(download_dir: Path) -> None:
    global _download_dir
    _download_dir = Path(download_dir)
    _download_dir.mkdir(parents=True, exist_ok=True)
    if str(_download_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(_download_dir))


def ensure(resource: str, package: str) -> None:
    """`resource` is the path NLTK looks up (e.g. corpora/treebank), `package` the download name."""
    if package in _checked:
        if _checked[package]:
            return
        raise DataUnavailable(f"NLTK data '{package}' is not installed and could not be downloaded.")
    try:
        nltk.data.find(resource)
    except LookupError:
        LOGGER.info("Downloading NLTK data: %s", package)
        try:
            nltk.download(package, download_dir=str(_download_dir) if _download_dir else None, quiet=True)
            nltk.data.find(resource)
        except Exception as exc:
            _checked[package] = False  # do not retry on every call
            raise DataUnavailable(
                f"NLTK data '{package}' is not installed and could not be downloaded ({exc}). "
                "Run: python scripts/download_nltk_data.py"
            ) from exc
    _checked[package] = True


REQUIRED = {
    "punkt_tab": "tokenizers/punkt_tab",
}
