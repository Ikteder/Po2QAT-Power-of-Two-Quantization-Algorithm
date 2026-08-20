from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import read_summary
from .experiment import ExperimentConfig, run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="po2qat",
        description="Run reproducible Po2 quantization-aware-training labs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="train, run Po2QAT, evaluate, and export weights")
    run.add_argument("--model", choices=["cnn", "vit", "llm", "all"], required=True)
    run.add_argument("--profile", choices=["smoke", "quick", "strong", "full"], default="quick")
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--bits", type=int, default=4)
    run.add_argument("--batch-size", type=int, help="profile default when omitted")
    run.add_argument("--learning-rate", type=float, default=3e-3)
    run.add_argument("--qat-learning-rate", type=float, default=5e-4)
    run.add_argument("--baseline-epochs", type=int)
    run.add_argument("--qat-epochs", type=int)
    run.add_argument("--baseline-steps", type=int)
    run.add_argument("--qat-steps", type=int)
    run.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    run.add_argument("--workers", type=int, default=0, help="0 is the safest cross-platform setting")
    inspect = subparsers.add_parser("inspect", help="print metrics and locate exported weight tables")
    inspect.add_argument("run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        print(read_summary(args.run_dir))
        return
    names = ["cnn", "vit", "llm"] if args.model == "all" else [args.model]
    for name in names:
        config = ExperimentConfig(
            model=name,
            profile=args.profile,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            num_bits=args.bits,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            qat_learning_rate=args.qat_learning_rate,
            baseline_epochs=args.baseline_epochs,
            qat_epochs=args.qat_epochs,
            baseline_steps=args.baseline_steps,
            qat_steps=args.qat_steps,
            device=args.device,
            workers=args.workers,
        )
        print(f"\n=== Running {name} ({args.profile}) ===", flush=True)
        run_dir = run_experiment(config)
        print(read_summary(run_dir), flush=True)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if not metrics.get("quality_gate_passed", False):
            print(
                "WARNING: the final Po2 model did not pass the configured quality gate. "
                "Review metrics.json and consider a longer QAT schedule.",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
