"""PDF and PPT parsing into Elements.

STATUS: written but NOT YET TESTED on real documents. It needs PyMuPDF and
python-pptx, which are installed in the lab (requirements-lab.txt). First real
run: lab session S2. The function names and return types are the interface the
rest of the code depends on, so they should not change.

OCR is a separate, pluggable step: `ocr` is any function image_path -> text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

from .elements import Element, Modality

OcrFn = Callable[[str], str]


def _missing(pkg: str, install: str) -> RuntimeError:
    return RuntimeError(f"{pkg} is not installed. In the lab run: pip install {install} (see docs/LAB_SETUP.md)")


def _configure_layout() -> None:
    """PyMuPDF 2.x can run an ONNX layout model inside find_tables(). On the lab CPU it was about 5 times slower
    (21.7 s against 4.6 s for a 70-page RBI PDF) and found fewer tables. It is off unless RAG_PYMUPDF_LAYOUT=1."""
    if os.environ.get("RAG_PYMUPDF_LAYOUT") == "1":
        return
    try:
        import pymupdf
        pymupdf.use_layout(False)
    except Exception:
        pass


def _table_rows(page, table, words=None) -> List[List[str]]:
    """Cell text rebuilt from the page's words, which are read once per page.

    table.extract() dropped spaces ("Home loan" -> "Homeloan") and split the decimal point onto its own line
    ("8.6" -> "86" plus a "." line), seen in lab session S2. A first fix called page.get_textbox() for every cell;
    that rescans the whole page each time and took 102 of 112 s on a 70-page RBI PDF (3,161 cells). Here a word
    belongs to the cell that contains its centre. Falls back to extract() if the table has no cell boxes.
    """
    try:
        if words is None:
            words = page.get_text("words")
        rows = []
        for row in table.rows:
            cells = []
            for box in row.cells:
                if box is None:
                    cells.append("")
                    continue
                x0, y0, x1, y1 = box
                inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
                inside.sort(key=lambda w: (w[5], w[6], w[7]))   # block, line, word number
                cells.append(" ".join(w[4] for w in inside))
            rows.append(cells)
        if rows:
            return rows
    except Exception:
        pass
    return table.extract()


def parse_pdf(path: str, doc_id: str, image_dir: str, ocr: Optional[OcrFn] = None,
              min_image_px: int = 80, metadata: Optional[dict] = None) -> List[Element]:
    """One Element per text block group, per table, and per image on each page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise _missing("PyMuPDF", "pymupdf") from None

    _configure_layout()
    meta = dict(metadata or {})
    out_dir = Path(image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    elements: List[Element] = []
    seen_images = set()  # an image used on several pages (logo, letterhead) is read once
    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            el = 0  # running ordinal per page: keeps ids of several images/tables on one page distinct
            text = page.get_text("text").strip()
            if text:
                elements.append(Element(doc_id, page_no, Modality.TEXT, text, metadata={**meta, "el": el}))
                el += 1

            tables = page.find_tables().tables
            words = page.get_text("words") if tables else None
            for table in tables:
                rows = _table_rows(page, table, words)
                body = "\n".join(" | ".join((c or "").strip() for c in row) for row in rows if row)
                if body.strip():
                    elements.append(Element(doc_id, page_no, Modality.TABLE, body, metadata={**meta, "el": el}))
                    el += 1

            for n, img in enumerate(page.get_images(full=True)):
                if img[0] in seen_images:
                    continue
                seen_images.add(img[0])
                pix = fitz.Pixmap(pdf, img[0])
                if pix.width < min_image_px or pix.height < min_image_px:
                    continue
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_path = str(out_dir / f"{doc_id}_p{page_no}_i{n}.png")
                pix.save(img_path)
                ocr_text = ocr(img_path) if ocr else ""
                elements.append(Element(doc_id, page_no, Modality.IMAGE, ocr_text,
                                        image_path=img_path, metadata={**meta, "el": el}))
                el += 1
    return elements


def parse_pptx(path: str, doc_id: str, image_dir: str, ocr: Optional[OcrFn] = None,
               metadata: Optional[dict] = None) -> List[Element]:
    """One Element per slide text, per table and per picture. Slide number is the page."""
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        raise _missing("python-pptx", "python-pptx") from None

    meta = dict(metadata or {})
    out_dir = Path(image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    elements: List[Element] = []
    for slide_no, slide in enumerate(Presentation(path).slides, start=1):
        texts = []
        el = 0  # running ordinal per slide (see parse_pdf)
        for n, shape in enumerate(slide.shapes):
            if shape.has_text_frame and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text.strip())
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                body = "\n".join(" | ".join(r) for r in rows)
                if body.strip():
                    elements.append(Element(doc_id, slide_no, Modality.TABLE, body, metadata={**meta, "el": el}))
                    el += 1
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_path = str(out_dir / f"{doc_id}_s{slide_no}_i{n}.{shape.image.ext}")
                Path(img_path).write_bytes(shape.image.blob)
                ocr_text = ocr(img_path) if ocr else ""
                elements.append(Element(doc_id, slide_no, Modality.IMAGE, ocr_text,
                                        image_path=img_path, metadata={**meta, "el": el}))
                el += 1
        if texts:
            elements.append(Element(doc_id, slide_no, Modality.TEXT, "\n".join(texts), metadata={**meta, "el": el}))
    return elements


def parse_document(path: str, doc_id: str, image_dir: str, ocr: Optional[OcrFn] = None,
                   metadata: Optional[dict] = None) -> List[Element]:
    """Pick the parser from the file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, doc_id, image_dir, ocr, metadata=metadata)
    if ext in (".pptx",):
        return parse_pptx(path, doc_id, image_dir, ocr, metadata=metadata)
    raise ValueError(f"unsupported file type {ext!r} (PDF and PPTX only; convert .ppt to .pptx first)")
