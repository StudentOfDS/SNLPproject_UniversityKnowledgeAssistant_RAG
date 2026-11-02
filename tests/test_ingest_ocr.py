import pytest

from app.ingest import extract_pages, validate_upload
from app.ocr import get_engine, lines_from_boxes
from tests.conftest import ATTENDANCE, image_only_pdf, make_image, make_pdf, needs_ocr, png_bytes


def test_digital_pdf_needs_no_ocr(settings):
    result = extract_pages("a.pdf", make_pdf(ATTENDANCE), settings, engine=None)
    assert len(result.pages) == 1
    assert not result.pages[0].ocr_used
    assert "75 percent" in result.pages[0].text


def test_scanned_page_is_reported_when_no_engine(settings):
    pdf = image_only_pdf(make_image("Scanned page with some words on it"))
    result = extract_pages("scan.pdf", pdf, settings, engine=None)
    assert result.pages == []
    assert result.skipped_ocr_pages == 1


def test_image_without_engine_gives_a_clear_error(settings):
    with pytest.raises(ValueError, match="OCR"):
        extract_pages("photo.png", png_bytes(make_image("hello")), settings, engine=None)


def test_validate_rejects_bad_files():
    with pytest.raises(ValueError, match="unsupported"):
        validate_upload("notes.docx", b"x", 5)
    with pytest.raises(ValueError, match="empty"):
        validate_upload("a.pdf", b"", 5)
    with pytest.raises(ValueError, match="valid PDF"):
        validate_upload("a.pdf", b"not a pdf", 5)
    with pytest.raises(ValueError, match="image"):
        validate_upload("a.png", b"not an image", 5)
    with pytest.raises(ValueError, match="larger"):
        validate_upload("a.pdf", b"%PDF" + b"0" * (2 * 1024 * 1024), 1)


def test_validate_accepts_real_files():
    validate_upload("a.pdf", make_pdf(ATTENDANCE), 5)
    validate_upload("a.png", png_bytes(make_image("hello")), 5)


def test_boxes_on_one_row_are_joined():
    box = lambda x, y: [[x, y], [x + 80, y], [x + 80, y + 20], [x + 80, y + 20]]
    items = [(box(10, 10), "Term End Exam", 0.9), (box(300, 12), "November 16", 0.9), (box(10, 60), "Re-exam", 0.9)]
    lines = lines_from_boxes(items).splitlines()
    assert lines[0] == "Term End Exam | November 16"
    assert lines[1] == "Re-exam"


def test_low_confidence_boxes_are_dropped():
    box = [[0, 0], [50, 0], [50, 20], [0, 20]]
    assert lines_from_boxes([(box, "noise", 0.1)]) == ""


@needs_ocr
def test_png_is_read_with_ocr(settings):
    engine = get_engine("auto")
    result = extract_pages("photo.png", png_bytes(make_image("Library fine is five rupees per day")), settings, engine)
    assert result.pages[0].ocr_used
    assert "library" in result.pages[0].text.lower()
    assert "rupees" in result.pages[0].text.lower()


@needs_ocr
def test_image_only_pdf_is_read_with_ocr(settings):
    engine = get_engine("auto")
    pdf = image_only_pdf(make_image("Hostel curfew is ten at night"))
    result = extract_pages("scan.pdf", pdf, settings, engine)
    assert result.pages[0].ocr_used
    assert "curfew" in result.pages[0].text.lower()
