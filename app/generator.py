"""Writes the answer from the retrieved passages: Gemini if a key is set, otherwise the best sentences quoted."""
from __future__ import annotations

import logging
import time

from app import answering
from app.models import Citation
from app.settings import Settings
from app.understanding import QueryAnalysis

LOGGER = logging.getLogger(__name__)

try:
    from google import genai
except ImportError:
    genai = None
    LOGGER.warning("google-genai is not installed, answers will be quoted from the documents.")

NOT_FOUND_MARKER = "NOT_FOUND"
_PLACEHOLDER_KEYS = {"", "your_actual_gemini_api_key_here"}


class GroundedAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None

        key = (settings.google_api_key or "").strip()
        if key.lower() in _PLACEHOLDER_KEYS:
            if key:
                LOGGER.warning("GOOGLE_API_KEY is still the placeholder from .env.example, so no LLM will be used.")
        elif genai is None:
            LOGGER.warning("GOOGLE_API_KEY is set but google-genai is not installed.")
        else:
            self.client = genai.Client(api_key=key)

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def generate(self, question: str, citations: list[Citation], analysis: QueryAnalysis,
                 history: list[dict] | None = None) -> str | None:
        """Return the answer text, or None if the passages do not contain the answer.

        Any LLM failure (bad key, quota, network, timeout) falls back to quoting the best sentences.
        """
        if not self.client:
            return answering.compose_answer(analysis, citations)[0] or None

        prompt = self._build_prompt(question, citations, history)
        started = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=self.settings.llm_model,
                contents=prompt,
                config={
                    "temperature": self.settings.llm_temperature,
                    "http_options": {
                        "timeout": int(self.settings.llm_timeout_seconds * 1000),  # milliseconds
                        "retry_options": {"attempts": self.settings.llm_max_attempts},
                    },
                },
            )
            answer = (response.text or "").strip()
            LOGGER.info("Gemini answered in %.1fs.", time.monotonic() - started)
        except Exception as exc:
            LOGGER.warning(
                "Gemini call failed after %.1fs (%s: %s). Quoting the best passage instead.",
                time.monotonic() - started, type(exc).__name__, exc,
            )
            return answering.compose_answer(analysis, citations)[0] or None

        if not answer:
            return answering.compose_answer(analysis, citations)[0] or None
        if answer.upper().startswith(NOT_FOUND_MARKER):
            return None
        return answer

    @staticmethod
    def _build_prompt(question: str, citations: list[Citation], history: list[dict] | None) -> str:
        earlier = ""
        turns = []
        for turn in (history or [])[-3:]:
            q, a = str(turn.get("question", "")).strip(), str(turn.get("answer", "")).strip()
            if q and a:
                turns.append(f"Q: {q}\nA: {a}")
        if turns:
            earlier = (
                "Earlier in this conversation (only use this to understand what the new question refers to):\n"
                + "\n\n".join(turns) + "\n\n"
            )

        passages = "\n\n".join(
            f"[{i}] {c.document_name}, page {c.page}, section {c.section}\n{c.text}"
            for i, c in enumerate(citations, start=1)
        )
        return (
            "You answer questions about university documents.\n"
            "Use only the passages below. Do not use outside knowledge and do not guess.\n"
            "Mention the passage number like [1] after each fact you use.\n"
            "If two passages disagree, say so.\n"
            f"If the passages do not contain the answer, reply with exactly {NOT_FOUND_MARKER} and nothing else.\n\n"
            f"{earlier}Passages:\n{passages}\n\nQuestion: {question}"
        )
