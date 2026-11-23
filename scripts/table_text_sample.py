"""Save raw table text (repr) from real PDFs so the table parser bug can be fixed.

    python scripts/table_text_sample.py [--docs 5] [--files D002 D010 ...]
Writes evidence/S3/table_text_raw_sample.txt
"""
import argparse
import glob
import re
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.elements import Modality  # noqa: E402
from rag.parsers import parse_pdf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5)
    ap.add_argument("--files", nargs="*")
    a = ap.parse_args()
    files = [str(ROOT / "data" / "raw" / f"{d}.pdf") for d in a.files] if a.files else sorted(glob.glob(str(ROOT / "data" / "raw" / "D*.pdf")))
    out, found = [], 0
    for f in files:
        doc = Path(f).stem
        try:
            els = parse_pdf(f, doc, tempfile.mkdtemp(), min_image_px=10**6)  # tables/text only
        except Exception as e:
            print(f"skip {doc}: {e}", file=sys.stderr)
            continue
        for e in els:
            if e.modality == Modality.TABLE and re.search(r"\d\.\d", e.text) and 80 < len(e.text) < 700:
                out.append(f"### {doc} page {e.page}\nprint(repr(table.text)) ->\n{e.text!r}\n")
                found += 1
                break
        if found >= a.docs:
            break
    dest = ROOT / "evidence" / "S3" / "table_text_raw_sample.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out) if out else "no table with a decimal number found")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
