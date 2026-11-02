"""Shared test setup. Tests run offline: no LLM key, keyword search only, fresh storage folders."""
from __future__ import annotations

import io
import os
import tempfile

_scratch = tempfile.mkdtemp(prefix="uka_tests_")
os.environ.update({
    "GOOGLE_API_KEY": "",
    "SEMANTIC_SEARCH": "false",
    "STORAGE_DIR": os.path.join(_scratch, "storage"),
    "DOCUMENTS_DIR": os.path.join(_scratch, "documents"),
    "AUTO_INDEX_ON_STARTUP": "false",
})

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.ocr import get_engine  # noqa: E402
from app.settings import Settings  # noqa: E402

needs_ocr = pytest.mark.skipif(get_engine("auto") is None, reason="no OCR engine installed")

ATTENDANCE = (
    "ATTENDANCE RULES\n"
    "Students must keep at least 75 percent attendance in every course. "
    "A student with lower attendance cannot appear for the semester end examination "
    "unless the dean approves a medical exemption. Attendance is counted from the first day of classes."
)
FEES = (
    "FEE PAYMENT\n"
    "Tuition fees are paid once per semester through the student portal. "
    "A late fee is charged when the payment is made after the due date. "
    "Refunds are only given when a student withdraws before the programme starts."
)


def make_pdf(*pages: str) -> bytes:
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return data


def make_image(text: str, size: int = 34) -> Image.Image:
    font = ImageFont.load_default(size=size)
    lines = text.split("\n")
    image = Image.new("RGB", (1500, 80 + len(lines) * (size + 22)), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * (size + 22)), line, fill="black", font=font)
    return image


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_only_pdf(image: Image.Image) -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=png_bytes(image))
    data = pdf.tobytes()
    pdf.close()
    return data


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(storage_dir=str(tmp_path / "storage"), documents_dir=str(tmp_path / "documents"),
                    google_api_key="", semantic_search=False)


@pytest.fixture
def kb(settings):
    from app.knowledge_base import KnowledgeBase
    return KnowledgeBase(settings)


@pytest.fixture
def policy_pdf() -> bytes:
    return make_pdf(ATTENDANCE, FEES)
