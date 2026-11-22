import unittest

from rag import ChunkConfig, Element, Modality, chunk_elements, chunk_table, chunk_text, count_tokens


class TestElement(unittest.TestCase):
    def test_valid_element_and_citation(self):
        e = Element("rbi_card_2024", 3, Modality.TEXT, "Liability is zero.")
        self.assertEqual(e.citation, "rbi_card_2024, p.3")
        self.assertEqual(e.element_id, "rbi_card_2024|p3|text|0")

    def test_rejects_bad_page_empty_text_and_missing_doc(self):
        with self.assertRaises(ValueError):
            Element("d", 0, Modality.TEXT, "x")
        with self.assertRaises(ValueError):
            Element("d", 1, Modality.TEXT, "   ")
        with self.assertRaises(ValueError):
            Element("", 1, Modality.TEXT, "x")

    def test_image_without_ocr_text_is_allowed_if_it_has_a_path(self):
        e = Element("d", 1, Modality.IMAGE, "", image_path="img.png")
        self.assertEqual(e.image_path, "img.png")


class TestChunkText(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("a b c", 10, 2), ["a b c"])

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(chunk_text("   ", 10, 2), [])

    def test_windows_overlap_and_cover_everything(self):
        words = [f"w{i}" for i in range(25)]
        chunks = chunk_text(" ".join(words), max_tokens=10, overlap=3)
        self.assertTrue(all(count_tokens(c) <= 10 for c in chunks))
        self.assertEqual(chunks[0].split()[-3:], chunks[1].split()[:3])  # overlap
        self.assertEqual(chunks[-1].split()[-1], "w24")                  # nothing dropped

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", 5, 5)
        with self.assertRaises(ValueError):
            ChunkConfig(max_tokens=0)


class TestChunkTable(unittest.TestCase):
    ROWS = [["Year", "GNPA"], ["2021", "7.3"], ["2022", "5.9"], ["2023", "3.9"], ["2024", "2.8"]]

    def test_header_repeated_in_every_chunk(self):
        chunks = chunk_table(self.ROWS, rows_per_chunk=2)
        self.assertEqual(len(chunks), 2)
        for c in chunks:
            self.assertTrue(c.startswith("Year | GNPA"))
        self.assertIn("2024 | 2.8", chunks[1])

    def test_header_only_and_empty(self):
        self.assertEqual(chunk_table([["A", "B"]]), ["A | B"])
        self.assertEqual(chunk_table([[""], [" "]]), [])


class TestChunkElements(unittest.TestCase):
    def test_modalities_are_chunked_and_numbered(self):
        text = Element("d", 1, Modality.TEXT, " ".join(f"t{i}" for i in range(30)), metadata={"doc_type": "circular"})
        table = Element("d", 2, Modality.TABLE, "Year | Val\n1 | 2\n3 | 4\n5 | 6")
        image = Element("d", 3, Modality.IMAGE, "bar chart of NPA", image_path="i.png")
        out = chunk_elements([text, table, image], ChunkConfig(max_tokens=10, overlap=2, table_rows_per_chunk=2))
        texts = [e for e in out if e.modality == Modality.TEXT]
        tables = [e for e in out if e.modality == Modality.TABLE]
        images = [e for e in out if e.modality == Modality.IMAGE]
        self.assertGreater(len(texts), 1)
        self.assertEqual([e.metadata["chunk"] for e in texts], list(range(len(texts))))
        self.assertEqual(texts[0].metadata["doc_type"], "circular")      # metadata survives
        self.assertEqual(len(tables), 2)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].image_path, "i.png")

    def test_image_without_text_is_kept_because_it_has_a_path(self):
        out = chunk_elements([Element("d", 1, Modality.IMAGE, "", image_path="i.png")])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
