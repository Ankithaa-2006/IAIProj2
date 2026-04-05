from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageMeta:
    code: str
    label: str
    nllb_code: str
    script: str


class LanguageRegistry:
    _languages = {
        "en": LanguageMeta("en", "English", "eng_Latn", "latin"),
        "hi": LanguageMeta("hi", "Hindi", "hin_Deva", "deva"),
        "kn": LanguageMeta("kn", "Kannada", "kan_Knda", "knda"),
        "ta": LanguageMeta("ta", "Tamil", "tam_Taml", "taml"),
        "ml": LanguageMeta("ml", "Malayalam", "mal_Mlym", "mlym"),
        "te": LanguageMeta("te", "Telugu", "tel_Telu", "telu"),
    }

    def supported(self) -> list[LanguageMeta]:
        return list(self._languages.values())

    def is_supported(self, code: str) -> bool:
        return code in self._languages

    def get(self, code: str) -> LanguageMeta:
        if code not in self._languages:
            raise KeyError(f"Unsupported language: {code}")
        return self._languages[code]

    def pair_label(self, source: str, target: str) -> str:
        return f"{self.get(source).label} -> {self.get(target).label}"

    def pair_key(self, source: str, target: str) -> str:
        return f"{source}:{target}"

    def model_id_for(self, source: str, target: str) -> str:
        return "facebook/nllb-200-distilled-600M"

