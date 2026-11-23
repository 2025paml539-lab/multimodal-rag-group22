"""Build corpus/manifest.csv from the public RBI Master Directions listing.

Downloads PDFs to data/raw/ (git-ignored), dedupes by SHA-256, counts pages and
writes one manifest row per distinct document.

    python scripts/build_manifest.py --limit 130
"""
import argparse
import csv
import hashlib
import html
import re
import sys
import time
from pathlib import Path

import requests
from pypdf import PdfReader

try:  # RBI serves an incomplete cert chain; use the OS trust store (Windows/macOS)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "corpus" / "manifest.csv"
LISTING = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Referer": "https://www.rbi.org.in/",
}
LICENSE = "Public RBI document, used for academic coursework"
FIELDS = ["doc_id", "title", "url", "type", "pages", "source", "sha256", "license_note"]


def listing_rows(page_html):
    """Yield (title, pdf_url) for each row that has a title link and a PDF link."""
    seen = set()
    for chunk in re.split(r"<tr\b", page_html, flags=re.I):
        t = re.search(
            r"""<a[^>]*href=["']?\s*(?:https?://[^/]+/Scripts/)?BS_ViewMasDirections\.aspx\?id=\d+[^>]*>\s*(.*?)</a>""",
            chunk,
            re.S | re.I,
        )
        p = re.search(r"href='(https://rbidocs\.rbi\.org\.in[^']+\.pdf)'", chunk, re.I)
        if not (t and p) or p.group(1) in seen:
            continue
        title = html.unescape(re.sub(r"<[^>]+>", "", t.group(1)))
        title = re.split(r"\s*\(Supersedes", title)[0]
        title = re.sub(r"\s+", " ", title).strip()
        seen.add(p.group(1))
        yield title, p.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=130, help="max listing rows to try")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    page = requests.get(LISTING, headers=UA, timeout=60).text
    rows = list(listing_rows(page))[: args.limit]
    print(f"listing rows to try: {len(rows)}")

    by_hash, out = {}, []
    for i, (title, url) in enumerate(rows, 1):
        tmp = RAW / f"_tmp_{i}.pdf"
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            tmp.write_bytes(r.content)
            sha = hashlib.sha256(r.content).hexdigest()
            if sha in by_hash:
                print(f"[{i}] duplicate of {by_hash[sha]}: {title[:60]}")
                tmp.unlink()
                continue
            pages = len(PdfReader(str(tmp)).pages)
        except Exception as e:  # keep going, report at the end
            print(f"[{i}] FAILED {url}: {e}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            continue
        doc_id = f"D{len(out) + 1:03d}"
        tmp.replace(RAW / f"{doc_id}.pdf")
        by_hash[sha] = doc_id
        out.append([doc_id, title, url, "pdf", pages, "RBI Master Directions", sha, LICENSE])
        time.sleep(0.3)

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(out)
    print(f"wrote {len(out)} distinct documents, {sum(r[4] for r in out)} pages -> {MANIFEST}")


if __name__ == "__main__":
    main()
