"""Short-term conversational memory and follow-up handling (experiment E4)."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from .llm import LLMClient

REWRITE_PROMPT = """Rewrite the last question so it can be understood without the conversation.
Keep names, numbers and document titles. Return only the rewritten question.

Conversation:
{history}

Last question: {question}

Rewritten question:"""


class ConversationMemory:
    """Keeps the last `max_turns` question/answer pairs. Nothing is stored between sessions."""

    def __init__(self, max_turns: int = 5):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._turns: Deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def history_text(self, max_answer_words: int = 40) -> str:
        """Answers are shortened so the memory costs few tokens."""
        lines: List[str] = []
        for q, a in self._turns:
            words = a.split()
            short = " ".join(words[:max_answer_words]) + (" ..." if len(words) > max_answer_words else "")
            lines.append(f"User: {q}\nAssistant: {short}")
        return "\n".join(lines)

    def standalone_question(self, question: str, llm: Optional[LLMClient]) -> str:
        """Turn a follow-up ('and for credit cards?') into a full question using the history.
        With no history, or if the rewrite fails or comes back empty, the original question is used."""
        if not self._turns or llm is None:
            return question
        try:
            result = llm.generate(
                REWRITE_PROMPT.format(history=self.history_text(), question=question),
                temperature=0.0, max_tokens=80,
            )
        except Exception:
            return question
        rewritten = result.text.strip().splitlines()[0].strip() if result.text.strip() else ""
        return rewritten or question
