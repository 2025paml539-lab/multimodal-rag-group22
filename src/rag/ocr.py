"""OCR for text inside images (charts, scanned tables). Returns an OcrFn: image_path -> str."""
from __future__ import annotations
import os
from typing import Callable

OcrFn = Callable[[str], str]


def _limit_threads() -> None:
    """RapidOCR creates its ONNX sessions with no thread limit. In a container that shares a large host, that can
    mean far more threads than the 6 CPUs we are allowed. RAG_ORT_THREADS sets the limit (default 4)."""
    n = int(os.environ.get("RAG_ORT_THREADS", "4"))
    if n <= 0:
        return
    try:
        from rapidocr_onnxruntime import utils
        original = utils.SessionOptions

        def limited():
            opts = original()
            opts.intra_op_num_threads = n
            opts.inter_op_num_threads = 1
            return opts

        utils.SessionOptions = limited
    except Exception:
        pass


def rapid_ocr() -> OcrFn:
    """RapidOCR wrapper. The model loads once. Any failure on one image returns "" instead of raising,
    so one bad image never stops ingestion. If RapidOCR is not installed, raises one clear error."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise RuntimeError("rapidocr-onnxruntime is not installed. In the lab: "
                           "pip install -r requirements-lab.txt") from e
    _limit_threads()
    engine = RapidOCR()

    def ocr(path: str) -> str:
        try:
            result, _ = engine(path)
            return " ".join(r[1] for r in result) if result else ""
        except Exception:
            return ""

    return ocr
