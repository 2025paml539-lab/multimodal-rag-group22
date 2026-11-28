"""Title context: a chunk is indexed together with its document title, so a question naming the document finds it."""
import unittest

from rag import Element, HashEmbedder, HybridIndex, Modality
from rag.index import searchable_text


def build(with_title):
    meta = (lambda t: {"title": t}) if with_title else (lambda t: {})
    els = [
        Element("D149", 1, Modality.TEXT, "Alpha Bank: Retail Loan Policy\nSynthetic document", metadata=meta("Alpha Bank: Retail Loan Policy")),
        Element("D149", 3, Modality.TABLE, "Category | Value\nPersonal loan | 10.8", metadata=meta("Alpha Bank: Retail Loan Policy")),
        Element("D150", 3, Modality.TABLE, "Category | Value\nPersonal loan | 13.7", metadata=meta("Bharat Union Bank: Retail Loan Policy")),
    ]
    idx = HybridIndex(HashEmbedder())
    idx.add(els)
    return idx


class TestTitleContext(unittest.TestCase):
    def test_searchable_text_prefixes_title_but_element_text_is_unchanged(self):
        e = Element("D", 3, Modality.TABLE, "a | b", metadata={"title": "Deck X"})
        self.assertEqual(searchable_text(e), "Deck X. a | b")
        self.assertEqual(e.text, "a | b")
        self.assertEqual(searchable_text(Element("D", 1, Modality.TEXT, "plain")), "plain")

    def test_question_naming_the_deck_finds_its_table_page(self):
        q = "In Alpha Bank: Retail Loan Policy, what is the interest rate for Personal loan?"
        top = build(True).search(q, k=1, mode="bm25")[0].element
        self.assertEqual((top.doc_id, top.page, top.modality), ("D149", 3, Modality.TABLE))

    def test_without_title_the_other_decks_table_cannot_be_told_apart(self):
        q = "In Alpha Bank: Retail Loan Policy, what is the interest rate for Personal loan?"
        hits = build(False).search(q, k=3, mode="bm25")
        self.assertNotEqual((hits[0].element.doc_id, hits[0].element.page), ("D149", 3))


if __name__ == "__main__":
    unittest.main()
