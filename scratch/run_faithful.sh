#!/usr/bin/env bash
cd "A:/omni read/.claude/worktrees/professional-manhwa-rendering-df6a32"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PYTHONIOENCODING=utf-8
PY="A:/omni read/.venv311/Scripts/python.exe"
"$PY" scratch/bareme.py score --run faithful --crops 2>&1
echo "=== FAITHFUL DONE rc=$? ==="
