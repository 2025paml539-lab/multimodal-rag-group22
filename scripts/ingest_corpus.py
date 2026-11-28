"""Ingest the whole corpus: parse -> OCR -> chunk -> embed -> save index.  Resumable.

Lab (session time is limited, so each document is cached the moment it is parsed):
    python scripts/ingest_corpus.py --embedder st --ocr
Offline check on a laptop:
    python scripts/ingest_corpus.py --docs smoke_docs --embedder hash --no-ocr --out /tmp/idx
Re-running skips documents already in the cache, so an interrupted session loses at most one document.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag import ChunkConfig, HashEmbedder, HybridIndex, chunk_elements  # noqa: E402
from rag.index import _element_from_dict, _element_to_dict  # noqa: E402
from rag.parsers import parse_document  # noqa: E402

CONFIGS = {"A": ChunkConfig(200, 40), "B": ChunkConfig(60, 10)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=[str(ROOT / "data" / "raw"), str(ROOT / "corpus" / "sample_decks")])
    ap.add_argument("--out", default=os.path.expanduser("~/rag_index"))
    ap.add_argument("--cache", default=os.path.expanduser("~/rag_cache"))
    ap.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--config", choices=list(CONFIGS), default="A")
    ap.add_argument("--ocr", dest="ocr", action="store_true")
    ap.add_argument("--no-ocr", dest="ocr", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (smoke run)")
    ap.set_defaults(ocr=False)
    a = ap.parse_args()

    files = sorted(str(p) for d in a.docs for p in Path(d).glob("*") if p.suffix.lower() in (".pdf", ".pptx"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print("no PDF or PPTX files in", a.docs)
        return 1
    cache = Path(a.cache)
    (cache / "parsed").mkdir(parents=True, exist_ok=True)
    ocr = None
    if a.ocr:
        from rag.ocr import rapid_ocr
        ocr = rapid_ocr()
    t0 = time.perf_counter()
    elements, failed = [], []
    print(f"PARSE {len(files)} files (cache {cache})", flush=True)
    for i, f in enumerate(files, 1):
        doc_id = Path(f).stem
        cached = cache / "parsed" / f"{doc_id}.json"
        if cached.exists():
            els = [_element_from_dict(d) for d in json.loads(cached.read_text(encoding="utf-8"))]
            tag = "cached"
        else:
            t = time.perf_counter()
            try:
                els = parse_document(f, doc_id, str(cache / "images" / doc_id), ocr)
            except Exception as e:  # one bad file must not stop the run
                failed.append((doc_id, repr(e)[:120]))
                print(f"  [{i}/{len(files)}] {doc_id}: FAILED {e!r:.100}", flush=True)
                continue
            cached.write_text(json.dumps([_element_to_dict(e) for e in els], ensure_ascii=False), encoding="utf-8")
            tag = f"{time.perf_counter() - t:.1f}s"
        elements += els
        print(f"  [{i}/{len(files)}] {doc_id}: {dict(collections.Counter(e.modality.name for e in els))} {tag}", flush=True)
    chunks = chunk_elements(elements, CONFIGS[a.config])
    print(f"CHUNK config {a.config}: {len(elements)} elements -> {len(chunks)} chunks", flush=True)
    if a.embedder == "st":
        from rag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder(a.st_model)
    else:
        emb = HashEmbedder()
    t = time.perf_counter()
    idx = HybridIndex(emb)
    idx.add(chunks)
    print(f"EMBED+INDEX {len(idx)} chunks in {time.perf_counter() - t:.1f}s", flush=True)
    idx.save(a.out)
    summary = {"files": len(files), "failed": failed, "elements": len(elements), "chunks": len(chunks),
               "config": a.config, "embedder": a.embedder, "ocr": a.ocr, "seconds": round(time.perf_counter() - t0, 1),
               "by_modality": dict(collections.Counter(c.modality.name for c in chunks))}
    Path(a.out, "ingest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAVED", a.out, json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
