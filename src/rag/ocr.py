"""OCR for text inside images (charts, scanned tables). Returns an OcrFn: image_path -> str."""
from __future__ import annotations
from typing import Callable

OcrFn = Callable[[str], str]


def rapid_ocr() -> OcrFn:
    """RapidOCR wrapper. The model loads once. Any failure on one image returns "" instead of raising,
    so one bad image never stops ingestion. If RapidOCR is not installed, raises one clear error."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise RuntimeError("rapidocr-onnxruntime is not installed. In the lab: "
                           "pip install -r requirements-lab.txt") from e
    engine = RapidOCR()

    def ocr(path: str) -> str:
        try:
            result, _ = engine(path)
            return " ".join(r[1] for r in result) if result else ""
        except Exception:
            return ""

    return ocr
