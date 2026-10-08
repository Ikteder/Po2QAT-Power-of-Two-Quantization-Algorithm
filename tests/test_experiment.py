from pathlib import Path
from types import SimpleNamespace

from po2qat import experiment


def test_run_directory_does_not_repeat_model_name() -> None:
    assert experiment.resolve_run_dir(Path("runs"), "cnn") == Path("runs/cnn")
    assert experiment.resolve_run_dir(Path("runs/cnn"), "cnn") == Path("runs/cnn")


def test_tpu_alias_resolves_through_torch_xla(monkeypatch) -> None:
    expected = SimpleNamespace(type="xla")
    fake_xla = SimpleNamespace(device=lambda: expected)
    monkeypatch.setattr(experiment.importlib, "import_module", lambda name: fake_xla)

    assert experiment.resolve_device("tpu") is expected
    assert experiment.resolve_device("xla") is expected


def test_xla_optimizer_step_synchronizes(monkeypatch) -> None:
    calls: list[str] = []
    optimizer = SimpleNamespace(step=lambda: calls.append("step"))
    fake_xla = SimpleNamespace(sync=lambda: calls.append("sync"))
    monkeypatch.setattr(experiment.importlib, "import_module", lambda name: fake_xla)

    experiment._optimizer_step(optimizer, SimpleNamespace(type="xla"))

    assert calls == ["step", "sync"]
