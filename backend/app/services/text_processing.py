from __future__ import annotations

import re


class TextPreprocessor:
    _multispace = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self._multispace.sub(" ", cleaned)
        cleaned = re.sub(r"\s+([,.!?;:।॥])", r"\1", cleaned)
        cleaned = re.sub(r"([,.!?;:।॥])(?=\w)", r"\1 ", cleaned)
        return cleaned.strip()

    def split_sentences(self, text: str) -> list[str]:
        segments = re.split(r"(?<=[.!?।॥])\s+", text.strip())
        return [segment.strip() for segment in segments if segment.strip()]
