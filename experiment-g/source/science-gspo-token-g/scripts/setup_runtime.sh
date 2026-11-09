#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo 'The pinned GPU runtime requires Linux x86_64 and Python 3.12.' >&2
  exit 1
fi
experiment_env="${SCIENCE_G_ENV:-/opt/science-g-env}"
python3.12 -m venv "$experiment_env"
"$experiment_env/bin/python" -m pip install --upgrade pip
"$experiment_env/bin/python" -m pip install -r requirements-runtime.txt
MAX_JOBS="${MAX_JOBS:-4}" NVCC_THREADS="${NVCC_THREADS:-2}" \
  "$experiment_env/bin/python" -m pip install --no-build-isolation -r requirements-kernels.txt
"$experiment_env/bin/python" -m pip install --no-deps -e .
"$experiment_env/bin/python" -m pip check
"$experiment_env/bin/python" -m science_gspo_g.cli inspect-config
printf 'Environment ready: %s\n' "$experiment_env"
