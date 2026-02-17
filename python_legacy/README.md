# Python Legacy Bridge (Phase 1)

Ce dossier contient le bridge FastAPI pour l'app desktop native.

## Objectif

- **Aucun modèle chargé au démarrage** (`lazy loading`)
- Traitement à la demande via API
- **Nettoyage mémoire strict** après traitement (`MemoryManager.cleanup_aggressive()`)

## Lancer le serveur

```powershell
uvicorn python_legacy.lazy_api:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`
- `POST /jobs` avec payload:

```json
{
  "input_path": "A:/omni read/image.png",
  "output_dir": "A:/omni read/output",
  "debug": false
}
```

- `GET /jobs/{job_id}?offset=0`

La UI Tauri peut créer un job puis faire du polling pour afficher la progression.
