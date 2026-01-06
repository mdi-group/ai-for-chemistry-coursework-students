#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -q -r requirements.txt

echo "✅ Colab environment ready."
