import os
import tempfile
import unittest
from pathlib import Path


class TestPdfImageDedupe(unittest.TestCase):
    def test_image_repeated_on_every_page_is_one_element(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("pymupdf not installed")
        from rag.parsers import parse_pdf
        d = tempfile.mkdtemp()
        png = Path(d, "logo.png")
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
        pix.clear_with(180)
        pix.save(str(png))
        pdf = pymupdf.open()
        for n in range(3):
            page = pdf.new_page()
            page.insert_text((50, 60), f"Page {n} text about bank loans")
            page.insert_image(pymupdf.Rect(50, 100, 250, 300), filename=str(png))
        path = os.path.join(d, "t.pdf")
        pdf.save(path)
        els = parse_pdf(path, "T", os.path.join(d, "img"), ocr=lambda p: "logo")
        images = [e for e in els if e.modality.value == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(len([e for e in els if e.modality.value == "text"]), 3)


if __name__ == "__main__":
    unittest.main()
