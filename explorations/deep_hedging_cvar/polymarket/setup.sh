#!/usr/bin/env bash
# CPU environment for the Polymarket market-making env (Python 3.13).
# Uses the same venv as the parent exploration (../.venv). PufferLib 3.0.0 pins
# numpy<2.0, which has no wheel for Python 3.13, so the pin is relaxed, along
# with the setup.py patch from ../setup.sh. NO_OCEAN=1 skips the bundled games.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv "${VENV:-.venv}"
source "${VENV:-.venv}/bin/activate"
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas pyarrow matplotlib setuptools wheel optuna cython
tmp=$(mktemp -d)
curl -sSL https://files.pythonhosted.org/packages/source/p/pufferlib/pufferlib-3.0.0.tar.gz | tar xz -C "$tmp"
sed -i "s/^c_extensions = \[\]$/c_extensions = []\nc_extension_paths = []/; s/'numpy<2.0'/'numpy'/" \
    "$tmp/pufferlib-3.0.0/setup.py"
NO_OCEAN=1 pip install --no-build-isolation "$tmp/pufferlib-3.0.0"
