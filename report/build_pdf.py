"""Build report/Part_A_report.pdf from report/REPORT_DRAFT.md.

    python report/build_pdf.py --mermaid <path to mermaid.min.js> --chrome <path to chrome.exe or msedge.exe>

The text is the markdown file. The markers {{HLD_DIAGRAM}} and {{LLD_SEQUENCE}} are replaced by the Mermaid diagrams in
docs/HLD.md and docs/LLD.md, so the report and the design docs cannot drift apart. Needs: pip install markdown.
"""
import argparse
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.3pt; line-height: 1.42; color: #111; }
h1 { font-size: 19pt; margin: 0 0 6px; } h2 { font-size: 14pt; margin: 18px 0 6px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 11.5pt; margin: 12px 0 4px; }
table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 9.2pt; }
th, td { border: 1px solid #999; padding: 3px 6px; text-align: left; vertical-align: top; } th { background: #eee; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
.mermaid { text-align: center; margin: 10px 0; page-break-inside: avoid; }
tr, img, pre { page-break-inside: avoid; }
"""


def mermaid_of(path):
    m = re.search(r"```mermaid\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    return m.group(1) if m else ""


def render_png(code, mermaid_js, chrome, width=1100):
    """Render one Mermaid diagram to a cropped PNG with headless Chrome and return its bytes."""
    from PIL import Image, ImageChops
    tmp = Path(tempfile.mkdtemp())
    page, shot = tmp / "d.html", tmp / "d.png"
    page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><script src='{Path(mermaid_js).resolve().as_uri()}'></script>"
                    "<script>mermaid.initialize({startOnLoad:true});</script></head><body style='margin:8px;background:#fff'>"
                    f"<div class='mermaid'>{code}</div></body></html>", encoding="utf-8")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--virtual-time-budget=20000",
                    f"--window-size={width},2600", "--force-device-scale-factor=2", f"--screenshot={shot}", page.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    im = Image.open(shot).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    im = im.crop((max(box[0] - 10, 0), max(box[1] - 10, 0), box[2] + 10, box[3] + 10)) if box else im
    out = tmp / "c.png"
    im.save(out)
    return out.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mermaid", required=True)
    ap.add_argument("--chrome", required=True)
    ap.add_argument("--out", default=str(ROOT / "report" / "Part_A_report.pdf"))
    ap.add_argument("--keep-html", action="store_true", help="keep report/_build.html for debugging")
    a = ap.parse_args()
    text = (ROOT / "report" / "REPORT_DRAFT.md").read_text(encoding="utf-8")
    boxes = {"{{HLD_DIAGRAM}}": mermaid_of(ROOT / "docs" / "HLD.md"), "{{LLD_SEQUENCE}}": mermaid_of(ROOT / "docs" / "LLD.md")}
    store = []
    for key, code in boxes.items():
        store.append(code)
        text = text.replace(key, f"\n\n@@MERMAID{len(store) - 1}@@\n\n")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    for i, code in enumerate(store):
        png = base64.b64encode(render_png(code, a.mermaid, a.chrome)).decode()
        body = body.replace(f"<p>@@MERMAID{i}@@</p>", f'<p style="text-align:center"><img style="max-width:100%;max-height:230mm" src="data:image/png;base64,{png}"></p>')
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
            f"</head><body>{body}</body></html>")
    tmp = ROOT / "report" / "_build.html"
    tmp.write_text(html, encoding="utf-8")
    out = Path(a.out)
    cmd = [a.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
           "--virtual-time-budget=30000", "--allow-file-access-from-files", tmp.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not a.keep_html:
        tmp.unlink(missing_ok=True)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
