#!/usr/bin/env sh
set -eu

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
printf '%s\n' 'Setup complete. Run: python -m po2qat run --model all --profile smoke --device cpu'

