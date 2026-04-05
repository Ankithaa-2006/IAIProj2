# Indian Multilingual Translation Project

FastAPI + React translation system for six languages:

- English (`en`)
- Hindi (`hi`)
- Kannada (`kn`)
- Tamil (`ta`)
- Malayalam (`ml`)
- Telugu (`te`)

The backend uses `google/translategemma-4b-it` directly for inference. Glossary-based translation fallback has been removed.

## Current Status

- Model download complete (safetensors shards present)
- Backend loads TranslateGemma successfully
- Frontend connected to backend
- All directed pairs validated: `30/30` passed

## Project Structure

- `backend/`: FastAPI service and translation pipeline
- `frontend/`: React + Vite UI

## UI

- Minimalist interface with light/dark mode toggle
- Clear source/target controls and candidate diagnostics
- Works on desktop and mobile layouts

## Run Backend

From repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/health -Method Get
```

Expected key signal in `model_status`:

- `loaded google/translategemma-4b-it in mode=translategemma-image-text-to-text ...`

## Run Frontend

If `node` is already on PATH:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

If Node is installed but not on PATH:

```powershell
$env:Path = 'C:\Program Files\nodejs;' + $env:Path
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Open:

- `http://127.0.0.1:5173`

## API Endpoints

- `GET /api/health`
- `GET /api/languages`
- `POST /api/translate`

Example translate request:

```json
{
	"text": "Good morning",
	"source_language": "en",
	"target_language": "ta",
	"max_candidates": 3
}
```

## Translation Pipeline Behavior

- Generates multiple candidates using decoding strategies (`greedy`, `beam`, `sample`, `strict`)
- Scores candidates using:
	- punctuation consistency
	- protected token preservation (emails/URLs/handles/numbers/title-cased tokens)
	- length consistency
	- target-script coverage
	- confidence estimate
- Selects highest-scoring candidate
- Uses retry with stricter profile when score is weak

## Pair Validation Summary

Validated in this workspace run:

- `total_pairs=30`
- `ok_pairs=30`
- `loaded_pairs=30`
- `failures=none`

## Notes

- The model currently runs on CPU in this environment; GPU will improve latency significantly.
- Terminal output may show garbled Indic characters due console encoding. API/UI output remains valid Unicode.

## Future Steps

1. Add GPU deployment profile with automatic `torch_dtype` and memory-aware generation settings.
2. Add server-side caching for repeated source-target pairs to reduce latency.
3. Add automated regression suite that runs all directed language pairs on every release.
4. Add translation quality dashboards (BLEU/chrF + human review snapshots).
5. Add containerized deployment (`Dockerfile` + production compose) for one-command startup.

## Model Weights and GitHub

The current model shards are very large. Pushing them directly into a normal GitHub repo is not practical due platform limits.

- Normal GitHub Git storage limit per file: 100 MB (hard block)
- Git LFS also has per-file and quota constraints, and these shards are multi-GB

Practical no-wait distribution options are:

1. Publish a prebuilt runtime image (container/VM) that already contains model weights.
2. Host weights on Hugging Face (current source of truth) and mirror to fast object storage/CDN near users.
3. Split and package model shards into release assets with reassembly step (more operational overhead).

If you want, the next pass can implement option 1 with a deployment image so users run immediately without a separate model download step.
