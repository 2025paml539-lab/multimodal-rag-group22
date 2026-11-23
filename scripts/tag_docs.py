"""Apply ontology.yaml rules to corpus/manifest.csv -> corpus/doc_metadata.csv.

    python scripts/tag_docs.py
"""
import csv
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ONT = yaml.safe_load((ROOT / "ontology.yaml").read_text(encoding="utf-8"))
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def first_match(rules, text, source=""):
    t = text.lower()
    for r in rules:
        if "source_contains" in r:
            if r["source_contains"].lower() in source.lower():
                return r["value"]
            continue
        if not r["match"] or any(m in t for m in r["match"]):
            return r["value"]


def find_date(title):
    m = re.search(rf"({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})", title)
    if m:
        from datetime import datetime

        return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date().isoformat()
    m = re.search(r"\b(20\d{2})(?:-\d{2})?\b", title)
    return m.group(1) if m else ""


def main():
    rows = list(csv.DictReader((ROOT / "corpus" / "manifest.csv").open(encoding="utf-8")))
    out = []
    for r in rows:
        title = r["title"]
        out.append({
            "doc_id": r["doc_id"],
            "doc_type": first_match(ONT["doc_type_rules"], title),
            "topic": first_match(ONT["topic_rules"], title),
            "source": r["source"].replace(" (PENDING APPROVAL)", ""),
            "date": find_date(title),
            "sensitivity": first_match(ONT["sensitivity_rules"], title, r["source"]),
        })
    dest = ROOT / "corpus" / "doc_metadata.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "documents tagged ->", dest)
    for k in ("doc_type", "topic", "sensitivity"):
        print(k, dict(Counter(o[k] for o in out).most_common()))
    print("with a date:", sum(1 for o in out if o["date"]))


if __name__ == "__main__":
    main()
