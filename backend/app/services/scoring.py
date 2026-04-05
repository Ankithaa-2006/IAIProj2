from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from ..core.schemas import CandidateScore


@dataclass(frozen=True)
class ScoredText:
    candidate_id: str
    strategy: str
    text: str
    confidence: float
    score: CandidateScore
    notes: list[str]


class HeuristicScorer:
    def score(self, source_text: str, candidate_text: str, source_language: str, target_language: str, confidence: float) -> CandidateScore:
        punctuation = self._punctuation_score(source_text, candidate_text)
        entities = self._entity_score(source_text, candidate_text)
        length = self._length_score(source_text, candidate_text)
        target_script = self._script_score(candidate_text, target_language)
        confidence_score = max(0.0, min(confidence, 1.0))
        total = round(
            0.22 * punctuation
            + 0.2 * entities
            + 0.18 * length
            + 0.28 * target_script
            + 0.12 * confidence_score,
            4,
        )
        return CandidateScore(
            punctuation=round(punctuation, 4),
            entities=round(entities, 4),
            length=round(length, 4),
            target_script=round(target_script, 4),
            confidence=round(confidence_score, 4),
            total=total,
        )

    def _punctuation_score(self, source_text: str, candidate_text: str) -> float:
        source = Counter(re.findall(r"[^\w\s]", source_text, flags=re.UNICODE))
        candidate = Counter(re.findall(r"[^\w\s]", candidate_text, flags=re.UNICODE))
        source_total = sum(source.values()) or 1
        difference = 0
        for punctuation, count in source.items():
            difference += abs(count - candidate.get(punctuation, 0))
        difference += sum(count for punctuation, count in candidate.items() if punctuation not in source)
        return max(0.0, 1.0 - min(1.0, difference / source_total))

    def _entity_score(self, source_text: str, candidate_text: str) -> float:
        source_entities = self._extract_protected_tokens(source_text)
        if not source_entities:
            return 0.75
        preserved = 0
        for entity in source_entities:
            if entity in candidate_text:
                preserved += 1
        return preserved / len(source_entities)

    def _extract_protected_tokens(self, source_text: str) -> list[str]:
        # Preserve emails, URLs, handles, hashtags, numerics, and title/upper-case words.
        patterns = [
            r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
            r"https?://\S+",
            r"(?:^|\s)[@#][A-Za-z0-9_]+",
            r"\b\d+(?:[.,]\d+)?\b",
            r"\b[A-Z][A-Za-z0-9_\-]{1,}\b",
            r"\b[A-Z]{2,}\b",
        ]
        protected: list[str] = []
        for pattern in patterns:
            protected.extend(match.strip() for match in re.findall(pattern, source_text))

        seen: set[str] = set()
        unique: list[str] = []
        for token in protected:
            lowered = token.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique.append(token)
        return unique

    def _length_score(self, source_text: str, candidate_text: str) -> float:
        source_length = max(len(source_text), 1)
        candidate_length = max(len(candidate_text), 1)
        ratio = min(source_length, candidate_length) / max(source_length, candidate_length)
        return max(0.0, min(1.0, ratio))

    def _script_score(self, candidate_text: str, target_language: str) -> float:
        letters = [char for char in candidate_text if char.isalpha()]
        if not letters:
            return 0.0
        if target_language == "en":
            latin = sum(1 for char in letters if char.isascii())
            return latin / len(letters)
        target_ranges = {
            "hi": ((0x0900, 0x097F),),
            "kn": ((0x0C80, 0x0CFF),),
            "ta": ((0x0B80, 0x0BFF),),
            "ml": ((0x0D00, 0x0D7F),),
            "te": ((0x0C00, 0x0C7F),),
        }
        ranges = target_ranges.get(target_language, ())
        if not ranges:
            return 0.5
        matches = 0
        for char in letters:
            codepoint = ord(char)
            if any(start <= codepoint <= end for (start, end) in ranges):
                matches += 1
        return matches / len(letters)


class CandidateSelector:
    def select(self, candidates: list[ScoredText]) -> ScoredText:
        return sorted(candidates, key=lambda item: (item.score.total, item.confidence), reverse=True)[0]
