$ErrorActionPreference = "Stop"

$PythonCommand = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
if ($PythonCommand -eq "py") {
    & py -3.12 -m venv .venv
} else {
    & python -m venv .venv
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -e ".[dev]"
& .venv\Scripts\python.exe -m pytest
Write-Host "Setup complete. Run: .venv\Scripts\python.exe -m po2qat run --model all --profile smoke --device cpu"

