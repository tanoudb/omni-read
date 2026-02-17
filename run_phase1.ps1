$ErrorActionPreference = "Stop"

Set-Location "A:\omni read"

$venvPython = "A:\omni read\.venv311\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Python venv introuvable: $venvPython"
}

Write-Host "[Phase1] Utilisation venv: $venvPython"

$port = 8000
$connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pidValue in $pids) {
        if ($pidValue -and $pidValue -ne 0) {
            try {
                Stop-Process -Id $pidValue -Force -ErrorAction Stop
                Write-Host "[Phase1] Processus arrêté sur port $port (PID: $pidValue)"
            } catch {
                Write-Warning "[Phase1] Impossible d'arrêter PID $pidValue : $($_.Exception.Message)"
            }
        }
    }
}

$env:LLM_REQUIRE_CUDA = "false"
$env:WEBTOON_LLM_REQUIRE_CUDA = "false"
$env:WEBTOON_TRANSLATION_BACKEND = "nllb"
$env:WEBTOON_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
$env:WEBTOON_AUTO_DETECT_SOURCE_LANG = "false"
$env:WEBTOON_TRANSLATION_NUM_BEAMS = "7"
$env:WEBTOON_TRANSLATION_MAX_LENGTH = "640"
$env:WEBTOON_MAX_GROUP_SIZE = "3"
$env:WEBTOON_CONTEXT_DISTANCE_THRESHOLD = "220"

Write-Host "[Phase1] LLM_REQUIRE_CUDA=false"
Write-Host "[Phase1] Translation backend=nllb (600M)"
Write-Host "[Phase1] Quality preset: auto-detect off, beams=7, max_length=640"
Write-Host "[Phase1] Démarrage FastAPI lazy en arrière-plan sur http://127.0.0.1:8000"

$apiProcess = Start-Process -FilePath $venvPython -ArgumentList "-m uvicorn python_legacy.lazy_api:app --host 127.0.0.1 --port 8000" -WorkingDirectory "A:\omni read" -PassThru
Write-Host "[Phase1] API PID: $($apiProcess.Id)"

$maxAttempts = 20
$serverReady = $false

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($health.status -eq "ok") {
            $serverReady = $true
            Write-Host "[Phase1] API prête (attempt $attempt/$maxAttempts)."
            break
        }
    } catch {
        Write-Host "[Phase1] Attente API... ($attempt/$maxAttempts)"
    }
}

if (-not $serverReady) {
    Write-Error "[Phase1] API non prête après $maxAttempts secondes."
}

$tauriDir = "A:\omni read\webtoon-translator-native"
if (-not (Test-Path $tauriDir)) {
    Write-Error "[Phase1] Dossier Tauri introuvable: $tauriDir"
}

$iconPath = Join-Path $tauriDir "src-tauri\icons\icon.ico"
if (-not (Test-Path $iconPath)) {
    Write-Host "[Phase1] icon.ico manquant -> auto-heal..."
    $iconScript = @"
from pathlib import Path
from PIL import Image
p = Path(r"$iconPath")
p.parent.mkdir(parents=True, exist_ok=True)
Image.new("RGBA", (256, 256), (45, 108, 223, 255)).save(p)
"@

    & $venvPython -c $iconScript

    if (-not (Test-Path $iconPath)) {
        Write-Error "[Phase1] Auto-heal icon échoué: $iconPath"
    }

    Write-Host "[Phase1] icon.ico créé automatiquement."
}

Write-Host "[Phase1] Lancement UI Tauri (dev)..."
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "Set-Location '$tauriDir'; npm run tauri:dev"

Write-Host "[Phase1] Démarrage automatique terminé."
Write-Host "[Phase1] API: http://127.0.0.1:8000"
