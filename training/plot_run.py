#!/usr/bin/env python3
"""Render local loss/learning-rate curves from a training run_record.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_record")
    parser.add_argument("--output")
    args = parser.parse_args()

    source = Path(args.run_record).resolve()
    if not source.exists():
        raise SystemExit(f"run record not found: {source}")
    record = json.loads(source.read_text(encoding="utf-8"))
    history = record.get("log_history")
    if not isinstance(history, list) or not history:
        raise SystemExit(f"no log_history in {source}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    training = [(row["step"], row["loss"]) for row in history if "step" in row and "loss" in row]
    validation = [(row["step"], row["eval_loss"]) for row in history if "step" in row and "eval_loss" in row]
    learning_rate = [(row["step"], row["learning_rate"]) for row in history if "step" in row and "learning_rate" in row]
    if not training and not validation:
        raise SystemExit(f"no train or validation loss values in {source}")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    if training:
        axes[0].plot(*zip(*training), label="Train loss", linewidth=1.8)
    if validation:
        axes[0].plot(*zip(*validation), marker="o", label="Validation loss", linewidth=1.8)
    axes[0].set_title(record.get("run_name", source.parent.name))
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if learning_rate:
        axes[1].plot(*zip(*learning_rate), color="tab:green", linewidth=1.8)
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(alpha=0.25)

    output = Path(args.output).resolve() if args.output else source.parent / "training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
