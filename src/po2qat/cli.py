from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import read_summary
from .experiment import ExperimentConfig, run_experiment


MODEL_CHOICES = {
    "1": "cnn",
    "2": "vit",
    "3": "llm",
    "cnn": "cnn",
    "vit": "vit",
    "llm": "llm",
}

PROFILE_CHOICES = {
    "1": "smoke",
    "2": "quick",
    "3": "strong",
    "4": "full",
    "smoke": "smoke",
    "quick": "quick",
    "strong": "strong",
    "full": "full",
}


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


def _ask_choice(prompt: str, choices: dict[str, str], input_fn=input, default: str | None = None) -> str:
    while True:
        answer = input_fn(prompt).strip().lower()
        if not answer and default is not None:
            return default
        if answer in choices:
            return choices[answer]
        print("Please enter one of the listed numbers or names.", flush=True)


def interactive_argv(input_fn=input) -> list[str]:
    print("\nPo2QAT interactive launcher", flush=True)
    print("Choose the model you want to run:", flush=True)
    print("  1. CNN - MobileNetTiny image classifier", flush=True)
    print("  2. ViT - Tiny Vision Transformer", flush=True)
    print("  3. LLM - TinyGPT character language model", flush=True)
    model = _ask_choice("Model [1/2/3]: ", MODEL_CHOICES, input_fn=input_fn)

    print("\nChoose an experiment profile:", flush=True)
    print("  1. smoke  - no download; checks that the pipeline works", flush=True)
    print("  2. quick  - short real-data classroom run (default)", flush=True)
    print("  3. strong - measured higher-quality reference schedule", flush=True)
    print("  4. full   - longest run using all available training data", flush=True)
    profile = _ask_choice("Profile [1/2/3/4, default 2]: ", PROFILE_CHOICES, input_fn=input_fn, default="quick")
    print(f"\nStarting {model} with the {profile} profile...", flush=True)
    return ["run", "--model", model, "--profile", profile]


def main(argv: list[str] | None = None) -> None:
    supplied = sys.argv[1:] if argv is None else argv
    if not supplied:
        supplied = interactive_argv()
    args = _parser().parse_args(supplied)
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
