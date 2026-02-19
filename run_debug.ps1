param(
    [string]$Image = "input/image.png",
    [string]$Output = "output/test_render",
    [string]$Mode = "hybrid",
    [switch]$NoDebug
)

Set-Location "A:/omni read"

$python = "A:/omni read/.venv311/Scripts/python.exe"
$args = @(
    "main.py",
    "--image", $Image,
    "--output", $Output,
    "--translation-mode", $Mode,
    "--log-level", "INFO"
)

if (-not $NoDebug) {
    $args += "--debug"
}

& $python @args
