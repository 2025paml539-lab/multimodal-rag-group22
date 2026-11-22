"""PDF and PPT parsing into Elements.

STATUS: written but NOT YET TESTED on real documents. It needs PyMuPDF and
python-pptx, which are installed in the lab (requirements-lab.txt). First real
run: lab session S2. The function names and return types are the interface the
rest of the code depends on, so they should not change.

OCR is a separate, pluggable step: `ocr` is any function image_path -> text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from .elements import Element, Modality

OcrFn = Callable[[str], str]


def _missing(pkg: str, install: str) -> RuntimeError:
    return RuntimeError(f"{pkg} is not installed. In the lab run: pip install {install} (see docs/LAB_SETUP.md)")


def parse_pdf(path: str, doc_id: str, image_dir: str, ocr: Optional[OcrFn] = None,
              min_image_px: int = 80, metadata: Optional[dict] = None) -> List[Element]:
    """One Element per text block group, per table, and per image on each page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise _missing("PyMuPDF", "pymupdf") from None

    meta = dict(metadata or {})
    out_dir = Path(image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    elements: List[Element] = []
    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                elements.append(Element(doc_id, page_no, Modality.TEXT, text, metadata=dict(meta)))

            for table in page.find_tables().tables:
                rows = table.extract()
                body = "\n".join(" | ".join((c or "").strip() for c in row) for row in rows if row)
                if body.strip():
                    elements.append(Element(doc_id, page_no, Modality.TABLE, body, metadata=dict(meta)))

            for n, img in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(pdf, img[0])
                if pix.width < min_image_px or pix.height < min_image_px:
                    continue
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_path = str(out_dir / f"{doc_id}_p{page_no}_i{n}.png")
                pix.save(img_path)
                ocr_text = ocr(img_path) if ocr else ""
                elements.append(Element(doc_id, page_no, Modality.IMAGE, ocr_text,
                                        image_path=img_path, metadata=dict(meta)))
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
        for n, shape in enumerate(slide.shapes):
            if shape.has_text_frame and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text.strip())
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                body = "\n".join(" | ".join(r) for r in rows)
                if body.strip():
                    elements.append(Element(doc_id, slide_no, Modality.TABLE, body, metadata=dict(meta)))
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_path = str(out_dir / f"{doc_id}_s{slide_no}_i{n}.{shape.image.ext}")
                Path(img_path).write_bytes(shape.image.blob)
                ocr_text = ocr(img_path) if ocr else ""
                elements.append(Element(doc_id, slide_no, Modality.IMAGE, ocr_text,
                                        image_path=img_path, metadata=dict(meta)))
        if texts:
            elements.append(Element(doc_id, slide_no, Modality.TEXT, "\n".join(texts), metadata=dict(meta)))
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
