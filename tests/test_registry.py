import csv
import tempfile
import unittest
from pathlib import Path

from rag import Element, Modality
from rag.registry import attach_metadata, load_registry

ROOT = Path(__file__).resolve().parent.parent


class TestRegistry(unittest.TestCase):
    def test_real_manifest_maps_every_local_deck_to_its_doc_id(self):
        reg = load_registry(str(ROOT / "corpus" / "manifest.csv"), str(ROOT / "corpus" / "doc_metadata.csv"))
        self.assertEqual(reg["SAMPLE_01_alpha_bank_loan_policy"]["doc_id"], "D149")
        self.assertEqual(reg["D146"]["doc_id"], "D146")
        self.assertEqual(reg["D146"]["doc_type"], "booklet_guide")
        self.assertEqual(reg["SAMPLE_20_eastern_bank_quarterly_results"]["sensitivity"], "synthetic")
        self.assertEqual(len(reg), 168)

    def test_attach_metadata_sets_doc_id_and_fields(self):
        e = Element("SAMPLE_01", 4, Modality.IMAGE, "chart", metadata={"el": 2})
        out = attach_metadata([e], {"doc_id": "D149", "doc_type": "slide_deck", "topic": "lending_credit", "sensitivity": ""})
        self.assertEqual(out[0].doc_id, "D149")
        self.assertEqual(out[0].metadata, {"el": 2, "doc_type": "slide_deck", "topic": "lending_credit"})

    def test_missing_files_give_empty_registry(self):
        self.assertEqual(load_registry("nope.csv", "nope2.csv"), {})


if __name__ == "__main__":
    unittest.main()
