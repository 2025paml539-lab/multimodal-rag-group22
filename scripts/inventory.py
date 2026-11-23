"""Corpus inventory: documents, pages/slides, tables, images per manifest row.

Counting rules match src/rag/parsers.py (tables from PyMuPDF find_tables, images
at least 80 px on both sides). Writes corpus/inventory.csv and
corpus/inventory_summary.json. Resumable: rows already in inventory.csv are kept.

    python scripts/inventory.py [--workers 6]
Raw PDFs are read from data/raw/<doc_id>.pdf; PPTX from the path in the manifest.
"""
import argparse
import csv
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.csv"
INV = ROOT / "corpus" / "inventory.csv"
SUMMARY = ROOT / "corpus" / "inventory_summary.json"
FIELDS = ["doc_id", "type", "pages", "text_pages", "tables", "images", "status"]
MIN_PX = 80


def count_pdf(path):
    import pymupdf

    pages = text_pages = tables = images = 0
    with pymupdf.open(path) as pdf:
        for page in pdf:
            pages += 1
            if page.get_text("text").strip():
                text_pages += 1
            tables += len(page.find_tables().tables)
            images += sum(1 for im in page.get_images(full=True) if im[2] >= MIN_PX and im[3] >= MIN_PX)
    return pages, text_pages, tables, images


def count_pptx(path):
    from pptx import Presentation

    prs = Presentation(path)
    slides = text_slides = tables = images = 0
    for s in prs.slides:
        slides += 1
        has_text = False
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                has_text = True
            if getattr(sh, "has_table", False) and sh.has_table:
                tables += 1
            if sh.shape_type == 13:
                images += 1
        text_slides += has_text
    return slides, text_slides, tables, images


def work(row):
    doc_id, typ, url = row["doc_id"], row["type"], row["url"]
    try:
        if url.startswith("corpus/"):
            p, tp, t, i = count_pptx(str(ROOT / url))
        else:
            p, tp, t, i = count_pdf(str(ROOT / "data" / "raw" / f"{doc_id}.pdf"))
        return [doc_id, typ, p, tp, t, i, "ok"]
    except Exception as e:  # report, keep going
        return [doc_id, typ, 0, 0, 0, 0, f"error: {type(e).__name__}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    done = {}
    if INV.exists():
        done = {r["doc_id"]: r for r in csv.DictReader(INV.open(encoding="utf-8")) if r["status"] == "ok"}
    todo = [r for r in rows if r["doc_id"] not in done]
    print(f"{len(rows)} documents, {len(done)} already counted, {len(todo)} to do", flush=True)

    results = {k: [v[f] for f in FIELDS] for k, v in done.items()}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, r): r["doc_id"] for r in todo}
        for n, f in enumerate(as_completed(futs), 1):
            res = f.result()
            results[res[0]] = res
            if n % 5 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
                _write(results)
    _write(results)
    _summary(rows, results)


def _write(results):
    with INV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(results[k] for k in sorted(results))


def _summary(rows, results):
    out = {"documents": len(results), "errors": sum(1 for r in results.values() if r[6] != "ok")}
    for label, pred in [("all", lambda r: True), ("pdf", lambda r: r[1] == "pdf"), ("ppt", lambda r: r[1] == "ppt")]:
        sel = [r for r in results.values() if pred(r)]
        out[label] = {
            "documents": len(sel),
            "pages_or_slides": sum(r[2] for r in sel),
            "documents_with_tables": sum(1 for r in sel if r[4] > 0),
            "tables": sum(r[4] for r in sel),
            "documents_with_images": sum(1 for r in sel if r[5] > 0),
            "images": sum(r[5] for r in sel),
        }
    SUMMARY.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
