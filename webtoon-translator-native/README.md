# Webtoon Translator Native - Phase 1

UI desktop minimale Tauri + bridge FastAPI lazy.

## Architecture

- `python_legacy/lazy_api.py` : API job-based, chargement modèles à la demande
- `webtoon-translator-native/` : shell desktop (UI simple RAM-friendly)

## Run (Phase 1)

1) Lancer API Python (depuis la racine projet):

```powershell
uvicorn python_legacy.lazy_api:app --host 127.0.0.1 --port 8000
```

2) Lancer UI Tauri (depuis `webtoon-translator-native`):

```powershell
npm install
npm run tauri:dev
```

## Contraintes mémoire prises en compte

- Pas de preload modèle au startup API
- Pipeline instancié par job en mode `lazy_models=True`
- `strict_memory_cleanup=True` + nettoyage agressif en fin de job
