"""Ingest the whole corpus: parse -> OCR -> chunk -> embed -> save index.  Resumable and parallel.

Lab (parsing is CPU work and the lab CPU is slow, so documents are parsed in several worker processes):
    python scripts/ingest_corpus.py --embedder st --ocr --workers 5
Offline check on a laptop:
    python scripts/ingest_corpus.py --docs smoke_docs --embedder hash --no-ocr --out /tmp/idx
Each document is cached the moment it is parsed. Re-running skips cached documents.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker: the lab shows 256 CPUs but allows 6
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag import ChunkConfig, HashEmbedder, HybridIndex, chunk_elements  # noqa: E402
from rag.index import _element_from_dict, _element_to_dict  # noqa: E402
from rag.parsers import parse_document  # noqa: E402
from rag.registry import attach_metadata, load_registry  # noqa: E402

CONFIGS = {"A": ChunkConfig(200, 40), "B": ChunkConfig(60, 10)}
_OCR = None


def _parse_one(task):
    """Runs in a worker process. Returns (index, doc_id, seconds or None if cached, error or None)."""
    global _OCR
    idx, path, doc_id, cache_dir, use_ocr = task
    cached = Path(cache_dir) / "parsed" / f"{doc_id}.json"
    if cached.exists():
        return idx, doc_id, None, None
    try:
        if use_ocr and _OCR is None:
            os.environ.setdefault("RAG_ORT_THREADS", "1")
            from rag.ocr import rapid_ocr
            _OCR = rapid_ocr()
        t = time.perf_counter()
        els = parse_document(path, doc_id, str(Path(cache_dir) / "images" / doc_id), _OCR)
        tmp = cached.with_suffix(".tmp")
        tmp.write_text(json.dumps([_element_to_dict(e) for e in els], ensure_ascii=False), encoding="utf-8")
        tmp.replace(cached)                      # a half-written file is never mistaken for a cached document
        return idx, doc_id, time.perf_counter() - t, None
    except Exception as e:                       # one bad file must not stop the run
        return idx, doc_id, None, repr(e)[:150]


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
    ap.add_argument("--workers", type=int, default=1, help="parallel parser processes (lab: 5)")
    ap.add_argument("--manifest", default=str(ROOT / "corpus" / "manifest.csv"))
    ap.add_argument("--doc-metadata", default=str(ROOT / "corpus" / "doc_metadata.csv"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (smoke run)")
    ap.set_defaults(ocr=False)
    a = ap.parse_args()

    files = sorted(str(p) for d in a.docs for p in Path(d).glob("*") if p.suffix.lower() in (".pdf", ".pptx"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print("no PDF or PPTX files in", a.docs)
        return 1
    registry = load_registry(a.manifest, a.doc_metadata)
    cache = Path(a.cache)
    (cache / "parsed").mkdir(parents=True, exist_ok=True)
    infos = [registry.get(Path(f).stem, {"doc_id": Path(f).stem}) for f in files]
    tasks = [(i, f, infos[i]["doc_id"], str(cache), a.ocr) for i, f in enumerate(files)]
    t0 = time.perf_counter()
    failed, done = [], 0
    print(f"PARSE {len(files)} files, {a.workers} workers (cache {cache})", flush=True)
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(_parse_one, t) for t in tasks]
            results = []
            for fut in as_completed(futures):
                results.append(fut.result())
                i, doc_id, secs, err = results[-1]
                done += 1
                print(f"  [{done}/{len(files)}] {doc_id}: " + (f"FAILED {err}" if err else "cached" if secs is None else f"{secs:.1f}s"), flush=True)
    else:
        results = []
        for t in tasks:
            results.append(_parse_one(t))
            i, doc_id, secs, err = results[-1]
            print(f"  [{len(results)}/{len(files)}] {doc_id}: " + (f"FAILED {err}" if err else "cached" if secs is None else f"{secs:.1f}s"), flush=True)
    failed = [(r[1], r[3]) for r in results if r[3]]
    elements = []
    for i, doc_id, _, err in sorted(results):
        if err:
            continue
        cached = cache / "parsed" / f"{doc_id}.json"
        els = [_element_from_dict(d) for d in json.loads(cached.read_text(encoding="utf-8"))]
        elements += attach_metadata(els, infos[i])
    print(f"PARSED in {time.perf_counter() - t0:.0f}s, {len(failed)} failed: {failed}", flush=True)
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
               "config": a.config, "embedder": a.embedder, "ocr": a.ocr, "workers": a.workers,
               "seconds": round(time.perf_counter() - t0, 1),
               "by_modality": dict(collections.Counter(c.modality.name for c in chunks))}
    Path(a.out, "ingest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAVED", a.out, json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
