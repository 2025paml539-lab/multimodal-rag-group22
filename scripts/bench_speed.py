"""30-second check of OCR and table speed, to run in the lab before the full ingest.
    python scripts/bench_speed.py
Prints seconds for one OCR call and for table detection on one real PDF, with the thread limit and layout settings used."""
import glob
import os
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
print("cpus visible:", os.cpu_count(), " affinity:", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a",
      " RAG_ORT_THREADS =", os.environ.get("RAG_ORT_THREADS", "4 (default)"))
from rag.ocr import rapid_ocr  # noqa: E402
img = (sorted(glob.glob(os.path.expanduser("~/rag_cache/images/*/*.png"))) or [None])[0]
if img:
    t = time.perf_counter(); ocr = rapid_ocr(); print(f"OCR load {time.perf_counter() - t:.1f}s")
    for i in range(3):
        t = time.perf_counter(); text = ocr(img); print(f"OCR call {i + 1}: {time.perf_counter() - t:.2f}s  ({len(text)} chars)")
else:
    print("no cached image found under ~/rag_cache/images")
pdf = str(ROOT / "data" / "raw" / "D002.pdf")
if os.path.exists(pdf):
    from rag.parsers import parse_pdf
    import tempfile
    t = time.perf_counter(); els = parse_pdf(pdf, "D002", tempfile.mkdtemp(), None, min_image_px=10 ** 6)
    print(f"D002 tables+text: {time.perf_counter() - t:.1f}s, {len(els)} elements")
