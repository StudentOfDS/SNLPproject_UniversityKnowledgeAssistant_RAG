"""Reading text out of PDFs and image files, using OCR where a page has no real text."""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageSequence

from app.ocr import OcrEngine
from app.settings import Settings

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS

# Above this much text a page is treated as a normal digital page and its pictures are ignored.
_TEXT_HEAVY_PAGE_CHARS = 1200
_MAX_IMAGES_PER_PAGE = 4


@dataclass
class PageText:
    number: int
    text: str
    ocr_used: bool = False


@dataclass
class Extraction:
    pages: list[PageText]
    skipped_ocr_pages: int = 0  # scan-like pages left unread because no OCR engine was available


def extension_of(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_upload(file_name: str, data: bytes, max_size_mb: int) -> None:
    extension = extension_of(file_name)
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"{file_name}: unsupported file type. Use one of: {allowed}.")
    if not data:
        raise ValueError(f"{file_name} is empty.")
    if len(data) > max_size_mb * 1024 * 1024:
        raise ValueError(f"{file_name} is larger than {max_size_mb} MB.")

    if extension == ".pdf":
        if not data.startswith(b"%PDF"):
            raise ValueError(f"{file_name} does not look like a valid PDF.")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise ValueError(f"{file_name} is password protected.")
                if pdf.page_count == 0:
                    raise ValueError(f"{file_name} has no pages.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{file_name} could not be opened as a PDF.") from exc
    else:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"{file_name} could not be opened as an image.") from exc


def extract_pages(file_name: str, data: bytes, settings: Settings, engine: OcrEngine | None) -> Extraction:
    if extension_of(file_name) == ".pdf":
        result = _pdf_pages(file_name, data, settings, engine)
    else:
        result = Extraction(pages=_image_pages(file_name, data, engine))
    result.pages = [page for page in result.pages if page.text.strip()]
    return result


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _image_pages(file_name: str, data: bytes, engine: OcrEngine | None) -> list[PageText]:
    if engine is None:
        raise ValueError(
            f"{file_name} is an image and no OCR engine is available. "
            "Install one with: pip install rapidocr-onnxruntime"
        )
    pages = []
    with Image.open(io.BytesIO(data)) as image:
        for number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            text = clean_text(engine.read(_png_bytes(frame)))
            pages.append(PageText(number=number, text=text, ocr_used=True))
    return pages


def _merge_ocr(existing: str, ocr_text: str) -> str:
    """Append OCR lines that are not already in the text layer."""
    def key(line: str) -> str:
        return re.sub(r"\W+", "", line.lower())

    seen = {key(line) for line in existing.splitlines()}
    extra = [line for line in ocr_text.splitlines() if len(key(line)) > 2 and key(line) not in seen]
    if not extra:
        return existing
    return (existing + "\n" + "\n".join(extra)).strip()


def _render(page: pymupdf.Page, dpi: int, clip: pymupdf.Rect | None = None) -> bytes:
    return page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")


def _large_image_boxes(page: pymupdf.Page, min_ratio: float) -> list[pymupdf.Rect]:
    page_area = page.rect.width * page.rect.height
    boxes = []
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if rect.is_empty or page_area <= 0:
            continue
        if rect.width * rect.height / page_area >= min_ratio:
            boxes.append(rect)
    boxes.sort(key=lambda r: r.width * r.height, reverse=True)
    return boxes[:_MAX_IMAGES_PER_PAGE]


def _pdf_pages(file_name: str, data: bytes, settings: Settings, engine: OcrEngine | None) -> Extraction:
    pages: list[PageText] = []
    ocr_pages = 0
    skipped_pages = 0
    dpi = max(settings.ocr_dpi, getattr(engine, "min_dpi", 0))

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text") or "")
            used_ocr = False

            if len(text) < settings.ocr_min_page_chars:
                if engine is None:
                    if page.get_image_info():
                        skipped_pages += 1
                else:
                    text = _merge_ocr(text, engine.read(_render(page, dpi)))
                    used_ocr = True
            elif engine is not None and settings.ocr_embedded_images and len(text) < _TEXT_HEAVY_PAGE_CHARS:
                for box in _large_image_boxes(page, settings.ocr_image_area_ratio):
                    new_text = _merge_ocr(text, engine.read(_render(page, dpi, clip=box)))
                    used_ocr = used_ocr or new_text != text
                    text = new_text

            ocr_pages += used_ocr
            pages.append(PageText(number=number, text=clean_text(text), ocr_used=used_ocr))

    if ocr_pages:
        LOGGER.warning(
            "%s: %d page(s) were read with OCR. OCR text can contain misread characters.",
            file_name, ocr_pages,
        )
    if skipped_pages:
        LOGGER.warning(
            "%s: %d page(s) look like scans but no OCR engine is installed, so they were skipped. "
            "Run: pip install rapidocr-onnxruntime",
            file_name, skipped_pages,
        )
    return Extraction(pages=pages, skipped_ocr_pages=skipped_pages)
