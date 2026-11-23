"""Append extra documents (PDF/PPT/PPTX) to corpus/manifest.csv.

New rows are marked "PENDING APPROVAL" in the source column until the instructor
approves the extra files. Existing SHA-256 hashes are skipped (dedupe).

    python scripts/add_docs.py --annual-report 20
    python scripts/add_docs.py --add "Title|https://host/file.pptx|Source name"
"""
import argparse
import csv
import hashlib
import html
import io
import re
import time
import zipfile
from pathlib import Path

import requests
from pypdf import PdfReader

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "corpus" / "manifest.csv"
AR_PAGE = "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Referer": "https://www.rbi.org.in/",
}
LICENSE = "Public document, used for academic coursework"
PENDING = " (PENDING APPROVAL)"


def annual_report_rows():
    t = requests.get(AR_PAGE, headers=UA, timeout=60).content.decode("utf-8", "replace")
    seen = set()
    for m in re.finditer(
        r"href='(https://rbidocs\.rbi\.org\.in/rdocs/AnnualReport/PDFs/[^']+)'[^>]*>\s*<img[^>]*alt='PDF\s*-\s*([^']*)'",
        t,
        re.I,
    ):
        title, url = html.unescape(m.group(2)).strip(), m.group(1)
        if url in seen or not title:
            continue
        seen.add(url)
        if "/0AR" in url or re.search(r"Letter of transmittal|Local Boards|Principal Officers|^Contents|Abbreviations", title):  # full report / front matter
            continue
        yield title if title.startswith("Annual Report") else f"Annual Report 2025-26: {title}", url, "RBI Annual Report 2025-26"


def count_units(data, kind):
    if kind == "pdf":
        return len(PdfReader(io.BytesIO(data)).pages)
    if kind == "pptx":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return len([n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)])
    return 0  # legacy .ppt: slide count unknown, fill manually


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annual-report", type=int, default=0, help="max Annual Report chapters to add")
    ap.add_argument("--add", action="append", default=[], help="'Title|URL|Source'")
    a = ap.parse_args()

    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    hashes = {r[6] for r in body}
    urls = {r[2] for r in body}
    next_id = max(int(r[0][1:]) for r in body) + 1

    todo = []
    if a.annual_report:
        todo += list(annual_report_rows())[: a.annual_report]
    for s in a.add:
        title, url, source = (s.split("|") + ["Public source"])[:3]
        todo.append((title, url, source))

    RAW.mkdir(parents=True, exist_ok=True)
    added = 0
    for title, url, source in todo:
        if url in urls:
            print("already in manifest:", title)
            continue
        try:
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            d = r.content
            kind = "pdf" if d.startswith(b"%PDF") else "pptx" if d.startswith(b"PK") and url.lower().endswith(".pptx") else "ppt" if url.lower().endswith(".ppt") and d[:4] == b"\xd0\xcf\x11\xe0" else None
            if not kind:
                raise ValueError("not a PDF/PPT/PPTX file")
            units = count_units(d, kind)
        except Exception as e:
            print(f"FAILED {title}: {e}")
            continue
        sha = hashlib.sha256(d).hexdigest()
        if sha in hashes:
            print("duplicate hash, skipped:", title)
            continue
        doc_id = f"D{next_id:03d}"
        (RAW / f"{doc_id}.{kind}").write_bytes(d)
        body.append([doc_id, title, url, "pdf" if kind == "pdf" else "ppt", units, source + PENDING, sha, LICENSE])
        hashes.add(sha)
        next_id += 1
        added += 1
        print(f"added {doc_id} {kind} {units:>4} {title[:70]}")
        time.sleep(0.3)

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    print(f"added {added}; manifest now {len(body)} rows")


if __name__ == "__main__":
    main()
