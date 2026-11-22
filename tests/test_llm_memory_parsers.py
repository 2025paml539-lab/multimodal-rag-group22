import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from rag import ConversationMemory, FakeLLMClient, OllamaClient
from rag.parsers import parse_document, parse_pdf, parse_pptx


class TestFakeLLM(unittest.TestCase):
    def test_scripted_responses_in_order_and_last_repeats(self):
        llm = FakeLLMClient(["one", "two"])
        self.assertEqual([llm.generate("p").text for _ in range(3)], ["one", "two", "two"])
        self.assertEqual(len(llm.calls), 3)

    def test_function_responses(self):
        llm = FakeLLMClient(lambda p: p.upper())
        self.assertEqual(llm.generate("abc").text, "ABC")


class _Stub(BaseHTTPRequestHandler):
    received = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.received = {"path": self.path, "body": body}
        payload = json.dumps({"response": " hello ", "prompt_eval_count": 12, "eval_count": 3}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestOllamaClient(unittest.TestCase):
    """Checks the request we send and how we read the reply, against a stub server (not a real model)."""

    def test_request_format_and_parsing(self):
        server = HTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = OllamaClient("qwen3:8b", host=f"http://127.0.0.1:{server.server_port}")
            result = client.generate("question?", temperature=0.0, seed=7, max_tokens=50)
        finally:
            server.shutdown()
            server.server_close()
        body = _Stub.received["body"]
        self.assertEqual(_Stub.received["path"], "/api/generate")
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertIs(body["think"], False)          # thinking off so answers are short and repeatable
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"]["seed"], 7)
        self.assertEqual(body["options"]["num_predict"], 50)
        self.assertEqual((result.text, result.input_tokens, result.output_tokens), ("hello", 12, 3))

    def test_unreachable_server_gives_a_helpful_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            OllamaClient("m", host="http://127.0.0.1:1", timeout=2).generate("x")
        self.assertIn("lab_start.sh", str(ctx.exception))


class TestMemory(unittest.TestCase):
    def test_keeps_only_the_last_turns(self):
        m = ConversationMemory(max_turns=2)
        for i in range(4):
            m.add(f"q{i}", f"a{i}")
        self.assertEqual(len(m), 2)
        self.assertNotIn("q0", m.history_text())
        self.assertIn("q3", m.history_text())

    def test_long_answers_are_shortened_in_history(self):
        m = ConversationMemory()
        m.add("q", " ".join(["w"] * 100))
        self.assertLess(len(m.history_text().split()), 60)

    def test_first_question_is_not_rewritten_and_costs_no_llm_call(self):
        llm = FakeLLMClient(["should not be used"])
        self.assertEqual(ConversationMemory().standalone_question("What is the limit?", llm), "What is the limit?")
        self.assertEqual(llm.calls, [])

    def test_follow_up_is_rewritten_using_history(self):
        m = ConversationMemory()
        m.add("What is the liability limit for card fraud?", "Zero if reported within 3 days.")
        llm = FakeLLMClient(["What is the liability limit for UPI fraud?"])
        out = m.standalone_question("And for UPI?", llm)
        self.assertEqual(out, "What is the liability limit for UPI fraud?")
        self.assertIn("card fraud", llm.calls[0]["prompt"])

    def test_failed_or_empty_rewrite_falls_back_to_the_original(self):
        m = ConversationMemory()
        m.add("q", "a")
        self.assertEqual(m.standalone_question("and then?", FakeLLMClient([""])), "and then?")

        class Broken:
            def generate(self, *a, **k):
                raise RuntimeError("down")
        self.assertEqual(m.standalone_question("and then?", Broken()), "and then?")


class TestParsers(unittest.TestCase):
    def test_unsupported_extension(self):
        with self.assertRaises(ValueError):
            parse_document("notes.docx", "d", "img")

    def test_missing_libraries_give_install_instructions(self):
        for fn, name in ((parse_pdf, "fitz"), (parse_pptx, "pptx")):
            try:
                __import__(name)
                self.skipTest(f"{name} is installed: real parsing is checked in the lab")
            except ImportError:
                with self.assertRaises(RuntimeError) as ctx:
                    fn("x", "d", "img")
                self.assertIn("LAB_SETUP", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
