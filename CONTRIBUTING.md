# Contributing

Changes should preserve the repository's teaching goals and cross-platform workflow.

1. Create a virtual environment and install `-e ".[dev]"`.
2. Add or update tests for algorithm and export changes.
3. Run `python -m pytest` on your platform.
4. Update the README and the relevant note, decision, dataset card, model card, or experiment log.
5. Do not commit downloaded datasets, virtual environments, or run artifacts.

Bug reports should include the command, operating system, Python/PyTorch versions, `config.json`, and the full error message. Do not upload large checkpoints unless a maintainer requests them.

