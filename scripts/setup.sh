#!/usr/bin/env sh
set -eu

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname -s)" = "Linux" ]; then
    torch_index_url="${PO2QAT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$torch_index_url"
fi
python -m pip install -e ".[dev]"
python -m pytest
printf '%s\n' 'Setup complete. Run: python -m po2qat run --model classroom --profile smoke --device cpu'

