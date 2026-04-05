from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core.schemas import HealthResponse, TranslationRequest, TranslationResponse
from .core.settings import settings
from .core.language import LanguageRegistry
from .services.pipeline import TranslationPipeline

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = TranslationPipeline()
registry = LanguageRegistry()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_status=pipeline.adapter.status,
        mode=pipeline.adapter.mode,
        safetensors=settings.use_safetensors,
        hf_transfer=settings.use_hf_transfer,
    )


@app.get("/api/languages")
def get_languages() -> dict[str, list[dict[str, str]]]:
    return {"languages": [{"code": item.code, "label": item.label, "nllb_code": item.nllb_code, "script": item.script} for item in registry.supported()]}


@app.post("/api/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest) -> TranslationResponse:
    try:
        return pipeline.translate(request)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
