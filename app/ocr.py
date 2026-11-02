"""OCR backends. RapidOCR installs with pip and works on Windows; Tesseract is used if it is on PATH."""
from __future__ import annotations

import io
import logging
import shutil
import statistics
import subprocess
import threading
from typing import Protocol

from PIL import Image

LOGGER = logging.getLogger(__name__)


class OcrEngine(Protocol):
    name: str
    min_dpi: int

    def read(self, image_bytes: bytes) -> str: ...


def lines_from_boxes(items: list[tuple[list, str, float]], min_score: float = 0.4) -> str:
    """Turn OCR boxes into text. Boxes on the same row are joined with ' | ' so table rows stay together."""
    boxes = []
    for points, text, score in items:
        text = (text or "").strip()
        if not text or float(score) < min_score:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        boxes.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "text": text})
    if not boxes:
        return ""

    typical_height = statistics.median(b["y1"] - b["y0"] for b in boxes) or 1
    boxes.sort(key=lambda b: (b["y0"] + b["y1"]) / 2)

    rows: list[list[dict]] = []
    for box in boxes:
        centre = (box["y0"] + box["y1"]) / 2
        if rows:
            last = rows[-1]
            last_centre = sum((b["y0"] + b["y1"]) / 2 for b in last) / len(last)
            if abs(centre - last_centre) <= typical_height * 0.6:
                last.append(box)
                continue
        rows.append([box])

    output = []
    for row in rows:
        row.sort(key=lambda b: b["x0"])
        parts = [row[0]["text"]]
        for previous, current in zip(row, row[1:]):
            gap = current["x0"] - previous["x1"]
            parts.append(" | " if gap > typical_height * 1.5 else " ")
            parts.append(current["text"])
        output.append("".join(parts))
    return "\n".join(output)


def _pad_thin_image(image_bytes: bytes, max_ratio: float = 5.0) -> bytes:
    """RapidOCR treats a very wide or tall strip as one line of text and skips detection.

    Adding white space around such an image makes it go through normal detection.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if max(width / height, height / width) <= max_ratio:
            return image_bytes
        side = max(width, height)
        canvas = Image.new("RGB", (side, side // 2), "white") if width >= height else Image.new("RGB", (side // 2, side), "white")
        canvas.paste(image, (0, 0))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()


class RapidOcrEngine:
    name = "rapidocr"
    min_dpi = 0

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # imported here so a missing install only matters when used

        self._engine = RapidOCR()
        self._lock = threading.Lock()

    def read(self, image_bytes: bytes) -> str:
        image_bytes = _pad_thin_image(image_bytes)
        with self._lock:
            result, _ = self._engine(image_bytes)
        return lines_from_boxes([(box, text, score) for box, text, score in (result or [])])


class TesseractEngine:
    name = "tesseract"
    min_dpi = 300  # Tesseract reads badly below this

    def __init__(self) -> None:
        self._binary = shutil.which("tesseract")
        if not self._binary:
            raise RuntimeError("tesseract is not on PATH")

    def read(self, image_bytes: bytes) -> str:
        completed = subprocess.run(
            [self._binary, "stdin", "stdout", "--psm", "4"],
            input=image_bytes, capture_output=True, timeout=180, check=False,
        )
        return completed.stdout.decode("utf-8", errors="replace").strip()


_engine_cache: dict[str, OcrEngine | None] = {}


def get_engine(preference: str = "auto") -> OcrEngine | None:
    """Return an OCR engine or None. The result is cached per preference."""
    preference = (preference or "auto").lower()
    if preference in _engine_cache:
        return _engine_cache[preference]

    engine: OcrEngine | None = None
    if preference != "off":
        candidates = {"auto": [RapidOcrEngine, TesseractEngine], "rapidocr": [RapidOcrEngine], "tesseract": [TesseractEngine]}
        for factory in candidates.get(preference, []):
            try:
                engine = factory()
                LOGGER.info("OCR engine: %s", engine.name)
                break
            except Exception as exc:
                LOGGER.debug("OCR engine %s not available: %s", factory.__name__, exc)
        if engine is None:
            LOGGER.warning(
                "No OCR engine available, scanned pages and images will be skipped. "
                "Run: pip install rapidocr-onnxruntime"
            )
    _engine_cache[preference] = engine
    return engine
