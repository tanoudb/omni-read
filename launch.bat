@echo off
echo Lancement du backend (API)...
start cmd /k "uvicorn python_legacy.lazy_api:app --host 127.0.0.1 --port 8000"

echo Lancement de l'application native (Tauri)...
cd webtoon-translator-native
start cmd /k "npm run tauri:dev"
