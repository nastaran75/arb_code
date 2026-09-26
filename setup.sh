#!/usr/bin/env bash
# Creates .venv with Python 3.11 and the pinned packages. Idempotent: if .venv already has the
# packages, this does nothing. Uses python3.11 if installed; otherwise uv (installed on demand,
# into ~/.local/bin, from https://astral.sh/uv) downloads a managed Python 3.11.
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ] && .venv/bin/python -c "import numpy, scipy, matplotlib" 2>/dev/null; then
    echo "setup: .venv is ready ($(.venv/bin/python --version))"; exit 0
fi
if command -v python3.11 >/dev/null 2>&1; then
    echo "setup: creating .venv with $(command -v python3.11)"
    python3.11 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "setup: python3.11 not found; installing uv to fetch a managed Python 3.11"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    echo "setup: creating .venv with uv-managed Python 3.11"
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
fi
echo "setup: .venv is ready ($(.venv/bin/python --version))"
