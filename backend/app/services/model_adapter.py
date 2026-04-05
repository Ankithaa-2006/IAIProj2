from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..core.language import LanguageRegistry
from ..core.settings import settings
from .text_processing import TextPreprocessor


class ModelAdapter:
    def __init__(self, registry: LanguageRegistry) -> None:
        self.registry = registry
        self.preprocessor = TextPreprocessor()
        self._status = "initializing; model inference path"
        self._model_bundle: Any | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def mode(self) -> str:
        return settings.model_mode

    def _token(self) -> str | None:
        token = os.getenv(settings.hf_token_env_var)
        if token:
            return token.strip()
        try:
            from huggingface_hub import get_token

            cached = get_token()
            return cached.strip() if cached else None
        except Exception:  # noqa: BLE001
            return None

    def _prepare_hf_runtime(self) -> None:
        Path(settings.hf_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.offload_dir).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = settings.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = settings.hf_cache_dir
        if settings.use_hf_transfer:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            os.environ["HF_HUB_DISABLE_XET"] = "1"

    def _has_local_artifacts(self, model_id: str) -> bool:
        from pathlib import Path

        from huggingface_hub import snapshot_download

        token = self._token()
        try:
            snapshot_path = snapshot_download(
                repo_id=model_id,
                cache_dir=settings.hf_cache_dir,
                token=token,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
                local_files_only=True,
            )
            snapshot_dir = Path(snapshot_path)
            required_files = {
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "config.json",
            }
            if not all((snapshot_dir / filename).exists() for filename in required_files):
                return False
            if any(snapshot_dir.glob("*.incomplete")):
                return False
            return True
        except Exception:  # noqa: BLE001
            return False

    def _load_model(self) -> Any:
        if self._model_bundle is not None:
            return self._model_bundle

        self._prepare_hf_runtime()
        model_id = settings.model_id
        if not settings.enable_model_download:
            self._status = (
                f"download disabled; model unavailable "
                f"(mode={settings.model_mode}, model={model_id})"
            )
            return None

        if settings.require_local_model_files and not self._has_local_artifacts(model_id):
            self._status = (
                f"model artifacts not ready locally for {model_id}; "
                "cannot run translation until artifacts are present"
            )
            return None

        try:
            if settings.model_mode == "translategemma-image-text-to-text":
                self._model_bundle = self._load_translategemma_bundle(model_id)
                return self._model_bundle
            raise ValueError(f"Unsupported model mode: {settings.model_mode}")
        except Exception as exc:  # noqa: BLE001
            self._status = (
                f"model load failure "
                f"(mode={settings.model_mode}, error={exc.__class__.__name__}: {exc})"
            )
            return None

    def _load_translategemma_bundle(self, model_id: str) -> Any:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        token = self._token()
        model_kwargs: dict[str, Any] = {
            "use_safetensors": settings.use_safetensors,
            "cache_dir": settings.hf_cache_dir,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
            "offload_folder": settings.offload_dir,
            "offload_state_dict": True,
        }
        if token:
            model_kwargs["token"] = token
        if settings.require_local_model_files:
            model_kwargs["local_files_only"] = True

        processor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
            cache_dir=settings.hf_cache_dir,
            local_files_only=settings.require_local_model_files,
        )
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        device = "cuda" if torch.cuda.is_available() else str(model.device)
        self._status = (
            f"loaded {model_id} in mode={settings.model_mode} on {device}; "
            f"safetensors={settings.use_safetensors}; hf_transfer={settings.use_hf_transfer}"
        )
        return {
            "kind": "translategemma",
            "processor": processor,
            "model": model,
        }

    def translate(self, text: str, source_language: str, target_language: str, strategy: str) -> tuple[str, float]:
        text = self.preprocessor.normalize(text)
        model_bundle = self._load_model()
        if model_bundle is None:
            raise RuntimeError(self._status)

        kind = model_bundle.get("kind")
        if kind == "translategemma":
            return self._translate_with_translategemma(model_bundle, text, source_language, target_language, strategy)

        self._status = f"model bundle failure (unsupported bundle kind={kind})"
        raise RuntimeError(self._status)

    def _translate_with_translategemma(
        self,
        model_bundle: dict[str, Any],
        text: str,
        source_language: str,
        target_language: str,
        strategy: str,
    ) -> tuple[str, float]:
        try:
            processor = model_bundle["processor"]
            model = model_bundle["model"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_language,
                            "target_lang_code": target_language,
                            "text": text,
                            "image": None,
                        },
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            generation_args: dict[str, Any] = {
                "max_new_tokens": 256,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if strategy == "beam":
                generation_args.update({"num_beams": 3, "repetition_penalty": 1.08})
            elif strategy == "sample":
                generation_args.update({"do_sample": True, "top_p": 0.92, "temperature": 0.75})
            elif strategy == "strict":
                generation_args.update({"num_beams": 4, "length_penalty": 1.0, "repetition_penalty": 1.1})
            else:
                generation_args.update({"num_beams": 1})

            output = model.generate(**inputs, **generation_args)
            prompt_tokens = inputs["input_ids"].shape[-1]
            generated = output.sequences[0][prompt_tokens:]
            decoded = processor.decode(generated, skip_special_tokens=True).strip()
            if not decoded:
                self._status = "model generation failure (empty decoded output)"
                raise RuntimeError(self._status)

            confidence = self._estimate_confidence(output.scores)
            return decoded, confidence
        except Exception as exc:  # noqa: BLE001
            self._status = (
                f"translategemma inference failure "
                f"(error={exc.__class__.__name__}: {exc})"
            )
            raise RuntimeError(self._status) from exc

    def _estimate_confidence(self, scores: list[Any]) -> float:
        try:
            import torch

            if not scores:
                return 0.5
            confidences: list[float] = []
            for step_scores in scores:
                probabilities = torch.softmax(step_scores[0], dim=-1)
                confidences.append(float(probabilities.max().item()))
            return max(0.05, min(0.98, sum(confidences) / len(confidences)))
        except Exception:  # noqa: BLE001
            return 0.5
