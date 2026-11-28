"""B1 tests: element_id uniqueness, index persistence, true scores per search mode, hosted LLM client."""
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from rag import (Element, HashEmbedder, HostedOpenAICompatibleClient, HybridIndex, Modality)


def two_images_on_one_page():
    return [
        Element("deck", 3, Modality.IMAGE, "zebra stripes chart quarterly revenue", image_path="a.png", metadata={"el": 0}),
        Element("deck", 3, Modality.IMAGE, "giraffe height chart annual cost", image_path="b.png", metadata={"el": 1}),
    ]


class TestElementIdCollision(unittest.TestCase):
    def test_two_images_on_one_page_have_distinct_ids(self):
        a, b = two_images_on_one_page()
        self.assertNotEqual(a.element_id, b.element_id)

    def test_both_survive_fusion_and_search_finds_the_first(self):
        idx = HybridIndex(HashEmbedder())
        idx.add(two_images_on_one_page())
        hits = idx.search("zebra stripes", k=5)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].element.image_path, "a.png")


class TestPersistence(unittest.TestCase):
    def test_save_load_round_trip_gives_identical_results(self):
        els = two_images_on_one_page() + [Element("doc", 1, Modality.TEXT, "liability is zero within three days", metadata={"el": 0})]
        idx = HybridIndex(HashEmbedder())
        idx.add(els)
        with tempfile.TemporaryDirectory() as d:
            idx.save(d)
            loaded = HybridIndex.load(d, HashEmbedder())
        self.assertEqual(len(loaded), len(idx))
        for q in ("zebra", "liability three days", "annual cost"):
            before = [(h.element.element_id, round(h.score, 9)) for h in idx.search(q, k=3)]
            after = [(h.element.element_id, round(h.score, 9)) for h in loaded.search(q, k=3)]
            self.assertEqual(before, after)
        self.assertEqual(loaded.bm25.elements[0].modality, Modality.IMAGE)


class TestSearchModeScores(unittest.TestCase):
    def setUp(self):
        self.idx = HybridIndex(HashEmbedder())
        self.idx.add(two_images_on_one_page())

    def test_bm25_and_dense_modes_return_real_scores_not_rrf(self):
        rrf_max = 1.0 / 61
        bm25 = self.idx.search("zebra stripes chart", k=2, mode="bm25")
        dense = self.idx.search("zebra stripes chart", k=2, mode="dense")
        self.assertGreater(bm25[0].score, rrf_max)
        self.assertTrue(-1.0 <= dense[0].score <= 1.0001)
        self.assertGreater(dense[0].score, rrf_max)

    def test_hybrid_still_uses_rank_fusion(self):
        hits = self.idx.search("zebra stripes chart", k=2, mode="hybrid")
        self.assertLessEqual(hits[0].score, 2.0 / 61 + 1e-9)


class _Chat(BaseHTTPRequestHandler):
    received = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Chat.received = {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        payload = json.dumps({"choices": [{"message": {"content": " hi "}}],
                              "usage": {"prompt_tokens": 9, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestHostedClient(unittest.TestCase):
    def test_request_and_parsing_against_stub_server(self):
        server = HTTPServer(("127.0.0.1", 0), _Chat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            c = HostedOpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}/v1", "gpt-oss-20b", "test-key-123")
            r = c.generate("question?", max_tokens=20, seed=5)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(_Chat.received["path"], "/v1/chat/completions")
        self.assertEqual(_Chat.received["auth"], "Bearer test-key-123")
        self.assertEqual(_Chat.received["body"]["model"], "gpt-oss-20b")
        self.assertEqual((r.text, r.input_tokens, r.output_tokens), ("hi", 9, 2))

    def test_missing_env_var_is_named_in_the_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                HostedOpenAICompatibleClient()
        self.assertIn("RAG_LLM_BASE_URL", str(ctx.exception))

    def test_key_is_not_in_error_text(self):
        c = HostedOpenAICompatibleClient("http://127.0.0.1:1/v1", "m", "SECRET-KEY-XYZ", timeout=2)
        with self.assertRaises(RuntimeError) as ctx:
            c.generate("x")
        self.assertNotIn("SECRET-KEY-XYZ", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
