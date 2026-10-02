#!/usr/bin/env bash
# CPU-only environment for this exploration.
# PufferLib 3.0.0 is built from source with only its PPO advantage kernel
# (NO_OCEAN=1 skips the bundled C games). Its setup.py references
# `c_extension_paths` before defining it on that path, so we patch it.
set -euo pipefail
python3 -m venv "${VENV:-.venv}"
source "${VENV:-.venv}/bin/activate"
pip install "torch==2.7.1" numpy matplotlib setuptools wheel
tmp=$(mktemp -d)
curl -sSL https://files.pythonhosted.org/packages/source/p/pufferlib/pufferlib-3.0.0.tar.gz | tar xz -C "$tmp"
sed -i 's/^c_extensions = \[\]$/c_extensions = []\nc_extension_paths = []/' "$tmp/pufferlib-3.0.0/setup.py"
NO_OCEAN=1 pip install --no-build-isolation "$tmp/pufferlib-3.0.0"
