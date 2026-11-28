import unittest

from rag import BM25Index, Element, HashEmbedder, HybridIndex, Hit, Modality, reciprocal_rank_fusion


def corpus():
    return [
        Element("rbi_fraud", 1, Modality.TEXT,
                "Customer liability for unauthorised electronic transactions is zero if reported within three days.",
                metadata={"doc_type": "circular", "topic": "fraud"}),
        Element("rbi_fraud", 2, Modality.TABLE,
                "Days to report | Liability\nwithin 3 days | zero\n4 to 7 days | transaction value limited",
                metadata={"doc_type": "circular", "topic": "fraud"}),
        Element("bank_deck", 7, Modality.IMAGE, "bar chart gross NPA ratio falling from 7.3 percent to 2.8 percent",
                image_path="npa.png", metadata={"doc_type": "presentation", "topic": "asset quality"}),
        Element("awareness", 4, Modality.TEXT, "Never share your card PIN or OTP with anyone, including bank staff.",
                metadata={"doc_type": "awareness", "topic": "fraud"}),
    ]


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.idx = BM25Index()
        self.idx.add(corpus())

    def test_finds_the_right_element(self):
        hits = self.idx.search("gross NPA ratio chart", k=3)
        self.assertEqual(hits[0].element.doc_id, "bank_deck")
        self.assertEqual(hits[0].rank, 1)

    def test_no_match_returns_nothing(self):
        self.assertEqual(self.idx.search("zzzz qqqq", k=3), [])

    def test_empty_index(self):
        self.assertEqual(BM25Index().search("anything"), [])

    def test_same_query_gives_identical_results_every_time(self):
        runs = [[h.element.element_id for h in self.idx.search("liability reported", k=4)] for _ in range(5)]
        self.assertTrue(all(r == runs[0] for r in runs))


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.idx = HybridIndex(HashEmbedder())
        self.idx.add(corpus())

    def test_metadata_filter_narrows_candidates(self):
        hits = self.idx.search("fraud", k=10, filters={"doc_type": "awareness"})
        self.assertTrue(hits)
        self.assertTrue(all(h.element.doc_id == "awareness" for h in hits))

    def test_filter_accepts_a_list_of_values(self):
        hits = self.idx.search("fraud", k=10, filters={"doc_type": ["circular", "awareness"]})
        self.assertTrue(all(h.element.metadata["doc_type"] in ("circular", "awareness") for h in hits))

    def test_modality_filter(self):
        hits = self.idx.search("NPA", k=10, modalities=[Modality.IMAGE])
        self.assertEqual([h.element.modality for h in hits], [Modality.IMAGE])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.idx.search("x", mode="magic")


class TestFusion(unittest.TestCase):
    def test_item_ranked_well_by_both_lists_wins(self):
        a, b, c = corpus()[:3]
        l1 = [Hit(a, 1.0, 1), Hit(b, 0.5, 2)]
        l2 = [Hit(c, 1.0, 1), Hit(b, 0.9, 2)]
        fused = reciprocal_rank_fusion([l1, l2], top=3)
        self.assertEqual(fused[0].element, b)         # appears in both lists
        self.assertEqual([h.rank for h in fused], [1, 2, 3])

    def test_hybrid_search_is_deterministic_and_finds_table(self):
        idx = HybridIndex(HashEmbedder())
        idx.add(corpus())
        runs = [[h.element.element_id for h in idx.search("liability reported within 3 days", k=3)] for _ in range(5)]
        self.assertTrue(all(r == runs[0] for r in runs))
        self.assertIn("rbi_fraud|p2|table|0|0", runs[0])
        self.assertEqual(len(idx), 4)

    def test_modes_return_hits(self):
        idx = HybridIndex(HashEmbedder())
        idx.add(corpus())
        for mode in ("hybrid", "bm25", "dense"):
            self.assertTrue(idx.search("card PIN OTP", k=2, mode=mode), mode)


if __name__ == "__main__":
    unittest.main()
