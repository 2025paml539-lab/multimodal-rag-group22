"""Generate 20 synthetic bank PPTX decks with known answers.

Each deck has 4 slides: title, a text rule, a table, and a bar-chart image.
Writes:
  corpus/sample_decks/SAMPLE_<nn>_<bank>_<topic>.pptx
  corpus/sample_decks/ground_truth.jsonl   (text, table and image facts per deck)
and appends manifest rows (source "Synthetic (generated)").

Deterministic (fixed seed, fixed zip timestamps). Files are committed so every
machine and the lab use identical bytes and the SHA-256 hashes stay valid.

    python scripts/make_sample_decks.py
"""
import csv
import hashlib
import io
import json
import random
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus" / "sample_decks"
MANIFEST = ROOT / "corpus" / "manifest.csv"
FIXED = datetime(2026, 1, 1, 0, 0, 0)
BANKS = ["Alpha Bank", "Bharat Union Bank", "Coastal Bank", "Deccan Bank", "Eastern Bank"]
TOPICS = {
    "loan_policy": dict(
        title="Retail Loan Policy",
        labels=["Home loan", "Personal loan", "Auto loan", "Education loan"],
        unit="% per annum", measure="interest rate", lo=7.0, hi=14.5, kind="dec"),
    "fraud_report": dict(
        title="Quarterly Fraud Report",
        labels=["Card fraud", "UPI fraud", "Net banking fraud", "ATM fraud"],
        unit="cases in Q3", measure="reported cases", lo=200, hi=1500, kind="int"),
    "kyc_guidelines": dict(
        title="KYC Guidelines",
        labels=["Low risk", "Medium risk", "High risk", "Politically exposed"],
        unit="months between KYC refreshes", measure="KYC refresh period", lo=6, hi=120, kind="int"),
    "quarterly_results": dict(
        title="Quarterly Results",
        labels=["Gross NPA", "Net NPA", "Net interest margin", "CASA ratio"],
        unit="% in Q3 FY26", measure="ratio", lo=0.5, hi=45.0, kind="dec"),
}


def fmt(v, kind):
    return f"{v:.1f}" if kind == "dec" else f"{int(v):,}"


def chart_png(title, labels, values, unit, kind):
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=100)
    bars = ax.bar(labels, values, color=["#2a6f97", "#468faf", "#61a5c2", "#89c2d9"])
    for b, v in zip(bars, values):
        ax.annotate(fmt(v, kind), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(unit, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def freeze_zip(path):
    """Rewrite the pptx zip with fixed timestamps so the bytes are reproducible."""
    with zipfile.ZipFile(path) as z:
        items = [(i.filename, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in items:
            zi = zipfile.ZipInfo(name, FIXED.timetuple()[:6])
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            z.writestr(zi, data)


def build(idx, bank, key, spec, rng):
    labels = spec["labels"]
    vals = [round(rng.uniform(spec["lo"], spec["hi"]), 1) if spec["kind"] == "dec" else rng.randint(int(spec["lo"]), int(spec["hi"])) for _ in labels]
    # make the top value unambiguous
    top = max(range(len(vals)), key=lambda i: vals[i])
    limit = rng.choice([25, 50, 75, 100])
    slug = bank.lower().replace(" ", "_")
    name = f"SAMPLE_{idx:02d}_{slug}_{key}.pptx"
    deck_title = f"{bank}: {spec['title']}"

    prs = Presentation()
    cp = prs.core_properties
    cp.author = cp.last_modified_by = "Synthetic corpus generator"
    cp.created = cp.modified = FIXED
    cp.title = deck_title

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = deck_title
    s.placeholders[1].text = "Synthetic document generated for BITS PCAM ZG516 coursework"

    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Approval rule"
    tf = s.placeholders[1].text_frame
    tf.text = f"{bank} requires a second approver for any {spec['title'].lower()} case above INR {limit} lakh."
    p = tf.add_paragraph()
    p.text = "Exceptions need written sign-off from the compliance head."

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = f"{spec['title']} table ({spec['unit']})"
    tbl = s.shapes.add_table(len(labels) + 1, 2, Inches(1), Inches(2), Inches(8), Inches(3)).table
    tbl.cell(0, 0).text, tbl.cell(0, 1).text = "Category", f"Value ({spec['unit']})"
    for r, (l, v) in enumerate(zip(labels, vals), 1):
        tbl.cell(r, 0).text, tbl.cell(r, 1).text = l, fmt(v, spec["kind"])
    for row in tbl.rows:
        for c in row.cells:
            c.text_frame.paragraphs[0].runs[0].font.size = Pt(16)

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Chart"
    png = chart_png(f"{deck_title} ({spec['unit']})", labels, vals, spec["unit"], spec["kind"])
    s.shapes.add_picture(io.BytesIO(png), Inches(1.5), Inches(1.8), width=Inches(7))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    prs.save(path)
    freeze_zip(path)

    facts = [
        dict(modality="text", question=f"In {deck_title}, above what amount in INR lakh is a second approver required?", answer=str(limit), page=2),
        dict(modality="table", question=f"In {deck_title}, what is the {spec['measure']} for {labels[1]}?", answer=fmt(vals[1], spec["kind"]), page=3),
        dict(modality="image", question=f"In the chart of {deck_title}, which category has the highest value?", answer=labels[top], page=4),
    ]
    return name, deck_title, path, facts


def main():
    rng = random.Random(2026)
    rows, truth = [], []
    idx = 1
    for key, spec in TOPICS.items():
        for bank in BANKS:
            name, title, path, facts = build(idx, bank, key, spec, rng)
            data = path.read_bytes()
            rows.append([title + "", f"corpus/sample_decks/{name}", "ppt", 4, "Synthetic (generated)",
                         hashlib.sha256(data).hexdigest(), "Synthetic, generated for coursework", name])
            for f in facts:
                truth.append(dict(f, file=name))
            idx += 1

    with MANIFEST.open(encoding="utf-8", newline="") as f:
        table = list(csv.reader(f))
    header, body = table[0], table[1:]
    body = [r for r in body if not r[2].startswith("corpus/sample_decks/")]  # idempotent
    nxt = max(int(r[0][1:]) for r in body) + 1
    id_by_file = {}
    for r in rows:
        did = f"D{nxt:03d}"
        id_by_file[r[7]] = did
        body.append([did] + r[:7])
        nxt += 1
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    with (OUT / "ground_truth.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, t in enumerate(truth, 1):
            t["id"] = f"SAMPLE-Q{i:03d}"
            t["doc_id"] = id_by_file[t["file"]]
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} decks, {len(truth)} facts; manifest now {len(body)} rows")


if __name__ == "__main__":
    main()
