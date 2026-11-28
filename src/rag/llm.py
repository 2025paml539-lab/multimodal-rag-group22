"""LLM client interface.

`FakeLLMClient` is used by every offline test. `OllamaClient` talks to the Ollama
server in the lab. It is tested here only against a stub HTTP server; the first
real call is lab session S2.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Union


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


class LLMClient(Protocol):
    def generate(self, prompt: str, **params: Any) -> LLMResult: ...


class FakeLLMClient:
    """Returns scripted answers. `responses` is a list (used in order, last one repeats)
    or a function prompt -> text. Every call is recorded in `.calls`."""

    def __init__(self, responses: Union[List[str], Callable[[str], str]] = ("OK",)):
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    def generate(self, prompt: str, **params: Any) -> LLMResult:
        self.calls.append({"prompt": prompt, "params": params})
        if callable(self._responses):
            text = self._responses(prompt)
        else:
            idx = min(len(self.calls) - 1, len(self._responses) - 1)
            text = self._responses[idx]
        return LLMResult(text, len(prompt.split()), len(text.split()), 0.0)


class OllamaClient:
    """Calls a local Ollama server. Thinking is switched off and a seed is sent so answers repeat."""

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 120.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def generate(self, prompt: str, temperature: float = 0.0, seed: int = 42,
                 max_tokens: int = 512, **extra: Any) -> LLMResult:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, **extra},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach Ollama at {self.host}: {e.reason}. "
                               "Run scripts/lab_start.sh first.") from e
        return LLMResult(
            text=data.get("response", "").strip(),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            seconds=time.perf_counter() - start,
        )


class HostedOpenAICompatibleClient:
    """Any hosted open-weight endpoint that speaks the OpenAI chat API (Groq, Together, OpenRouter,
    Fireworks, HF Inference). Fallback for gpt-oss-20b, which the lab's Ollama 0.12.6 cannot pull (HTTP 412).
    Settings come only from environment variables. The key is never logged or written to a file:
        export RAG_LLM_BASE_URL=https://.../v1   RAG_LLM_MODEL=openai/gpt-oss-20b   RAG_LLM_API_KEY=...
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or self._need("RAG_LLM_BASE_URL")).rstrip("/")
        self.model = model or self._need("RAG_LLM_MODEL")
        self._api_key = api_key or self._need("RAG_LLM_API_KEY")
        self.timeout = timeout

    @staticmethod
    def _need(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"environment variable {name} is not set (see docs/LAB_SETUP.md)")
        return value

    def generate(self, prompt: str, temperature: float = 0.0, seed: int = 42,
                 max_tokens: int = 512, **extra: Any) -> LLMResult:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "seed": seed, "max_tokens": max_tokens, **extra}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"hosted LLM returned HTTP {e.code} from {self.base_url}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach hosted LLM at {self.base_url}: {e.reason}") from None
        usage = data.get("usage") or {}
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return LLMResult(text.strip(), int(usage.get("prompt_tokens", 0)),
                         int(usage.get("completion_tokens", 0)), time.perf_counter() - start)
