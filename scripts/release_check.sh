#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[release-check] cleaning dist/"
rm -rf dist

echo "[release-check] building sdist + wheel"
uv build

echo "[release-check] creating temp venv"
TMP_VENV="$(mktemp -d)/venv"
python -m venv "$TMP_VENV"
source "$TMP_VENV/bin/activate"

echo "[release-check] installing built wheel"
pip install --upgrade pip >/dev/null
pip install dist/*.whl >/dev/null

echo "[release-check] import smoke test"
python - <<'PY'
from features_goldmine import GoldenFeatures
print("GoldenFeatures import OK:", GoldenFeatures.__name__)
PY

echo "[release-check] done"
