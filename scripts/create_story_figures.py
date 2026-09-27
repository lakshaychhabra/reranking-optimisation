#!/usr/bin/env python3
"""Create report-ready figures from frozen dataset, training, and eval summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


INK = "#17202A"
MUTED = "#68737D"
GRID = "#DDE3E8"
BLUE = "#246BCE"
TEAL = "#148A8A"
GREEN = "#2F855A"
ORANGE = "#D97706"
RED = "#C2413B"
PURPLE = "#7656A8"
PALETTE = [BLUE, TEAL, GREEN, ORANGE, PURPLE, RED]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def clean_axis(axis: Axes, *, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)


def add_note(figure: Figure, text: str) -> None:
    figure.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=8, color=MUTED)


def save_figure(figure: Figure, output_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def label_bars(axis: Axes, *, percent: bool = False, padding: float = 3) -> None:
    labels = [f"{bar.get_height():.0%}" if percent else f"{bar.get_height():,.0f}" for bar in axis.patches]
    axis.bar_label(axis.containers[0], labels=labels, padding=padding, color=INK, fontsize=9)


def plot_dataset_funnel_and_splits(
    quoted: dict[str, Any], splits: dict[str, Any], teacher: dict[str, Any]
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [0.9, 1.35]})
    figure.suptitle("A deliberately separated data pipeline", x=0.02, ha="left", fontsize=18, fontweight="bold")

    funnel = quoted["collection_funnel"]
    values = [funnel["attempts"], funnel["successes"], teacher["annotation_count"]]
    labels = ["Generated & priced", "Quoteable scenarios", "Teacher-labelled"]
    colors = ["#B8C8D8", BLUE, TEAL]
    axis = axes[0]
    bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.58)
    axis.set_xlim(0, max(values) * 1.18)
    axis.set_title("Collection funnel", loc="left")
    axis.set_xlabel("Scenarios")
    clean_axis(axis, grid_axis="x")
    for bar, value in zip(bars, values[::-1]):
        axis.text(value + max(values) * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", color=INK, fontweight="bold")
    axis.text(
        funnel["successes"] / 2,
        1.0,
        f"{funnel['success_rate_pct']:.1f}% quote success",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )

    split_counts = {
        "Reward\ncalibration": splits["hierarchy"]["stage_1"]["reward_calibration"],
        "SFT\ntrain": splits["hierarchy"]["stage_2"]["sft_pool"]["sft_train"],
        "SFT\nvalidation": splits["hierarchy"]["stage_2"]["sft_pool"]["sft_validation"],
        "GRPO\ntrain": splits["hierarchy"]["stage_2"]["grpo_pool"]["grpo_train"],
        "GRPO\nvalidation": splits["hierarchy"]["stage_2"]["grpo_pool"]["grpo_validation"],
    }
    axis = axes[1]
    x = np.arange(len(split_counts))
    bars = axis.bar(x, split_counts.values(), color=[PURPLE, BLUE, "#83AEE8", TEAL, "#80C7C7"], width=0.68)
    axis.set_xticks(x, split_counts.keys())
    axis.set_ylabel("Scenarios")
    axis.set_title("Every scenario has exactly one role", loc="left")
    axis.set_ylim(0, 960)
    clean_axis(axis)
    axis.bar_label(bars, fmt="{:,.0f}", padding=3, color=INK, fontsize=9, fontweight="bold")
    axis.text(
        0.98,
        0.995,
        "Golden benchmark: 50 held out\n0 used for training or tuning",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=RED,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": RED, "boxstyle": "round,pad=0.4"},
    )
    add_note(figure, "Source: frozen quoted-data, split, and teacher manifests. Split counts are disjoint and exhaustive (n=1,490).")
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    return figure


def plot_dataset_distribution(splits: dict[str, Any], quoted: dict[str, Any]) -> Figure:
    distribution = splits["source_distribution"]
    panels = [
        ("age_stratum", "Building age"),
        ("hazard_tier", "Hazard tier"),
        ("building_type", "Building type"),
        ("construction_class", "Construction class"),
        ("owner_occupied", "Owner occupied"),
        ("natural_hazard_requested", "Natural hazard cover"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    figure.suptitle("The synthetic portfolio is broad—and the splits preserve it", x=0.02, ha="left", fontsize=18, fontweight="bold")
    for axis, (key, title) in zip(axes.flat, panels):
        items = sorted(distribution[key].items(), key=lambda item: item[1]["pct"], reverse=True)
        labels = [item[0].replace("-", " ").replace("true", "Yes").replace("false", "No") for item in items]
        values = [item[1]["pct"] / 100 for item in items]
        y = np.arange(len(labels))
        bars = axis.barh(y, values, color=PALETTE[: len(values)], height=0.62)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.22)
        axis.set_title(title, loc="left")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        clean_axis(axis, grid_axis="x")
        for bar, value in zip(bars, values):
            axis.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color=INK, fontsize=9)
    quote_stats = quoted["dataset"]
    add_note(
        figure,
        f"Source: split manifest. n={quote_stats['successful_scenarios']:,} scenarios, {quote_stats['total_quotes']:,} quotes, "
        f"{quote_stats['unique_insurers']} insurers, {quote_stats['unique_products']} products. Largest final-split marginal deviation: 1.56 pp.",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    return figure


def plot_reward_calibration(calibration: dict[str, Any]) -> Figure:
    baseline = calibration["baseline"]["metrics"]
    selected = calibration["selected"]["metrics"]
    metrics = [
        ("top1_accuracy", "Top-1 agreement"),
        ("top3_recall", "Top-3 recall"),
        ("pairwise_accuracy", "Pairwise accuracy"),
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    figure.suptitle("Reward calibration makes the scorer look more like the teacher", x=0.02, ha="left", fontsize=18, fontweight="bold")
    x = np.arange(len(metrics))
    width = 0.34
    baseline_values = [baseline[key] for key, _ in metrics]
    selected_values = [selected[key] for key, _ in metrics]
    first = axis.bar(x - width / 2, baseline_values, width, label="Supplied reward", color="#AAB5BF")
    second = axis.bar(x + width / 2, selected_values, width, label="Calibrated reward", color=PURPLE)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_ylabel("Agreement on calibration set")
    clean_axis(axis)
    axis.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005))
    axis.bar_label(first, labels=[f"{value:.1%}" for value in baseline_values], padding=3, color=MUTED)
    axis.bar_label(second, labels=[f"{value:.1%}" for value in selected_values], padding=3, color=INK, fontweight="bold")
    for idx, (before, after) in enumerate(zip(baseline_values, selected_values)):
        axis.text(idx, max(before, after) + 0.075, f"+{(after - before) * 100:.1f} pp", ha="center", color=PURPLE, fontweight="bold")
    add_note(figure, "Source: 200-scenario reward-calibration split only; 5-fold selection, no golden or GRPO-validation data. Not directly comparable to golden metrics.")
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    return figure


def plot_sft_training(record: dict[str, Any]) -> Figure:
    history = record["log_history"]
    train = [(row["step"], row["loss"]) for row in history if "loss" in row]
    validation = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
    lr = [(row["step"], row["learning_rate"]) for row in history if "learning_rate" in row]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    figure.suptitle("SFT converges quickly on a small, clean target", x=0.02, ha="left", fontsize=18, fontweight="bold")
    axis = axes[0]
    axis.plot(*zip(*train), color=BLUE, marker="o", linewidth=2.2, label="Train loss")
    axis.plot(*zip(*validation), color=ORANGE, marker="o", linewidth=2.2, label="Validation loss")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Train and validation loss", loc="left")
    clean_axis(axis)
    axis.legend(frameon=False)
    best_step, best_loss = min(validation, key=lambda pair: pair[1])
    axis.annotate(
        f"Best validation\n{best_loss:.4f} at step {best_step}",
        xy=(best_step, best_loss),
        xytext=(-85, 36),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=INK,
    )
    axis = axes[1]
    axis.plot(*zip(*lr), color=GREEN, marker="o", linewidth=2.2)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Learning rate")
    axis.set_title("Warm-up and cosine decay", loc="left")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    clean_axis(axis)
    add_note(
        figure,
        f"Source: frozen SFT run record. 300 train / 90 validation examples; 3 epochs; {record['elapsed_seconds'] / 60:.1f} min elapsed; "
        f"{record['peak_gpu_memory_gb']:.1f} GB peak GPU memory.",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.91))
    return figure


def plot_grpo_training(record: dict[str, Any]) -> Figure:
    rows = [row for row in record["log_history"] if "reward" in row]
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    figure.suptitle("The 10-step GRPO continuation is stable—but nearly saturated", x=0.02, ha="left", fontsize=18, fontweight="bold")

    axis = axes[0]
    series = [
        ("reward", "Total reward", BLUE),
        ("rewards/calibrated_ranking_reward/mean", "Ranking", TEAL),
        ("rewards/calibrated_top1_reward/mean", "Top-1", PURPLE),
    ]
    for key, label, color in series:
        axis.plot(steps, [row[key] for row in rows], marker="o", linewidth=2, label=label, color=color)
    axis.set_ylim(0.94, 1.005)
    axis.set_xlabel("GRPO step")
    axis.set_ylabel("Mean reward")
    axis.set_title("Reward components", loc="left")
    clean_axis(axis)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    axis.plot(steps, [row["kl"] for row in rows], marker="o", linewidth=2.2, color=ORANGE)
    axis.set_xlabel("GRPO step")
    axis.set_ylabel("KL divergence")
    axis.set_title("Policy drift falls", loc="left")
    clean_axis(axis)

    axis = axes[2]
    lengths = [row["completion_length"] for row in rows]
    valid_json = [row["rewards/strict_json_reward/mean"] for row in rows]
    axis.plot(steps, lengths, marker="o", linewidth=2.2, color=GREEN, label="Completion tokens")
    axis.set_xlabel("GRPO step")
    axis.set_ylabel("Completion length (tokens)")
    axis.set_ylim(min(lengths) - 0.5, max(lengths) + 0.5)
    axis.set_title("Output format stays stable", loc="left")
    clean_axis(axis)
    right = axis.twinx()
    right.plot(steps, valid_json, marker="s", linestyle="--", linewidth=1.8, color=PURPLE, label="Strict JSON")
    right.set_ylabel("Strict-JSON reward", color=PURPLE)
    right.set_ylim(0.98, 1.005)
    right.tick_params(axis="y", colors=PURPLE)
    right.spines["top"].set_visible(False)
    add_note(
        figure,
        f"Source: frozen GRPO run record. 810 train / 90 validation examples; 10 optimizer steps; {record['elapsed_seconds'] / 60:.1f} min elapsed; "
        f"{record['peak_gpu_memory_gb']:.1f} GB peak GPU memory. Only three logged train observations—read as diagnostics, not a smooth trend.",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.91))
    return figure


def metric_summary(path: Path) -> dict[str, float]:
    return read_json(path)["metrics"]


def plot_model_progress(repo: Path, calibration: dict[str, Any]) -> Figure:
    base = metric_summary(repo / "artifacts/training/evaluations/golden-base-qwen35-2b-700tok/summary.json")
    sft = metric_summary(repo / "artifacts/training/evaluations/golden-sft-qwen35-2b-seed3407-700tok/summary.json")
    grpo = metric_summary(repo / "artifacts/training/evaluations/golden-grpo-qwen35-2b-10step-700tok/summary.json")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.55, 1]})
    figure.suptitle("SFT delivers the step-change; GRPO confirms the plateau", x=0.02, ha="left", fontsize=18, fontweight="bold")

    stages = ["2B base", "+ SFT", "+ 10-step GRPO"]
    x = np.arange(len(stages))
    axis = axes[0]
    for key, label, color in [
        ("top1_accuracy", "Top-1 agreement", BLUE),
        ("top3_recall", "Top-3 recall", TEAL),
        ("reward_ratio", "Reward ratio", ORANGE),
        ("strict_json_rate", "Strict JSON", PURPLE),
    ]:
        values = [row[key] for row in (base, sft, grpo)]
        axis.plot(x, values, marker="o", markersize=7, linewidth=2.2, label=label, color=color)
        for idx, value in enumerate(values):
            axis.text(idx, value + (0.025 if key != "strict_json_rate" else -0.055), f"{value:.0%}", ha="center", color=color, fontsize=8)
    axis.set_xticks(x, stages)
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_ylabel("Held-out golden metric")
    axis.set_title("Model progression on the same 50 scenarios", loc="left")
    clean_axis(axis)
    axis.legend(frameon=False, ncol=2, loc="lower right", fontsize=8)

    axis = axes[1]
    baseline = calibration["baseline"]["metrics"]
    selected = calibration["selected"]["metrics"]
    names = ["Top-1", "Top-3", "Pairwise"]
    before = [baseline["top1_accuracy"], baseline["top3_recall"], baseline["pairwise_accuracy"]]
    after = [selected["top1_accuracy"], selected["top3_recall"], selected["pairwise_accuracy"]]
    y = np.arange(3)
    for idx, (left, right) in enumerate(zip(before, after)):
        axis.plot([left, right], [idx, idx], color=GRID, linewidth=5, solid_capstyle="round")
        axis.scatter(left, idx, color="#98A5B1", s=70, zorder=3, label="Supplied" if idx == 0 else None)
        axis.scatter(right, idx, color=PURPLE, s=70, zorder=3, label="Calibrated" if idx == 0 else None)
        axis.text(left - 0.025, idx, f"{left:.0%}", va="center", ha="right", color=MUTED, fontsize=8)
        axis.text(right + 0.025, idx, f"{right:.0%}", va="center", ha="left", color=PURPLE, fontsize=8, fontweight="bold")
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlim(0.35, 1.02)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_title("Reward calibration (separate 200-scenario split)", loc="left")
    axis.set_xlabel("Teacher agreement")
    clean_axis(axis, grid_axis="x")
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(0.04, 0.25))
    add_note(figure, "Left: frozen golden summaries; the golden set was evaluated only after choices were frozen. Right: calibration split, shown separately because its scores are not comparable to golden.")
    figure.tight_layout(rect=(0, 0.07, 1, 0.91))
    return figure


def serving_cost_assumption(sweep: dict[str, Any], sustained: dict[str, Any]) -> dict[str, Any]:
    low_traffic = next(row for row in sweep["levels"] if row["concurrency"] == 1)
    saturated = next(row for row in sustained["levels"] if row["concurrency"] == 64)
    tokens_per_recommendation = saturated["mean_prompt_tokens"] + saturated["mean_completion_tokens"]
    costs = saturated["cost_eur_per_1k_at_utilization"]

    def eur_per_mtok(cost_per_1k: float) -> float:
        return cost_per_1k * 1_000 / tokens_per_recommendation

    return {
        "method": "measured vLLM serving benchmark on a RunPod RTX 4090",
        "gpu_hourly_usd": saturated["gpu_hourly_usd"],
        "usd_per_eur": saturated["usd_per_eur"],
        "workload_tokens_per_recommendation": round(tokens_per_recommendation, 3),
        "recommended_planning_utilization_pct": 25,
        "price_assumption_eur_per_mtok": round(eur_per_mtok(costs["25_pct"]), 6),
        "cost_eur_per_1k_recommendations": {
            "saturated_100_pct": costs["100_pct"],
            "fleet_50_pct": costs["50_pct"],
            "fleet_25_pct_planning": costs["25_pct"],
            "fleet_10_pct": costs["10_pct"],
            "single_request_no_batching": low_traffic["saturated_cost_eur_per_1k"],
        },
        "effective_eur_per_mtok": {
            "saturated_100_pct": round(eur_per_mtok(costs["100_pct"]), 6),
            "fleet_50_pct": round(eur_per_mtok(costs["50_pct"]), 6),
            "fleet_25_pct_planning": round(eur_per_mtok(costs["25_pct"]), 6),
            "fleet_10_pct": round(eur_per_mtok(costs["10_pct"]), 6),
            "single_request_no_batching": round(
                eur_per_mtok(low_traffic["saturated_cost_eur_per_1k"]), 6
            ),
        },
        "measured_capacity": {
            "requests_per_second_at_concurrency_64": saturated["requests_per_second"],
            "p50_latency_seconds": saturated["latency_seconds"]["p50"],
            "p95_latency_seconds": saturated["latency_seconds"]["p95"],
            "mean_gpu_utilization_pct": saturated["telemetry"]["gpu_utilization_pct"]["mean"],
            "strict_json_rate": saturated["strict_json_rate"],
            "failed_requests": saturated["failed_requests"],
        },
        "gpu_memory": {
            "total_gib": 23.52,
            "weights_plus_non_torch_gib": 3.91,
            "peak_activation_gib": 0.40,
            "cuda_graph_gib": 0.14,
            "kv_cache_gib": 16.85,
            "max_engine_copies_before_kv_cache": 5,
            "practical_independent_instances_with_1_gib_kv_each": 4,
            "recommended_topology": "one continuously batched engine; replicas only for isolation or availability",
        },
        "scope": sweep["cost_scope"],
    }


def plot_cost_quality(
    summary: dict[str, Any], cost_assumption: dict[str, Any]
) -> Figure:
    included = [model for model in summary["models"] if model.get("cost_per_1k_recommendations_eur") is not None]
    gpt_sol = next(model for model in summary["models"] if model["model"] == "gptsol")
    gpt_sol_eur = {
        **gpt_sol,
        "cost_per_1k_recommendations_eur": float(gpt_sol["cost_per_1k_recommendations_usd"])
        / float(cost_assumption["usd_per_eur"]),
    }
    included.append(gpt_sol_eur)
    local = [model for model in summary["models"] if model["model"] in {"qwen35_2b_sft", "qwen35_2b_grpo"}]
    costs = cost_assumption["cost_eur_per_1k_recommendations"]
    low = float(costs["saturated_100_pct"])
    high = float(costs["single_request_no_batching"])
    midpoint = (low + high) / 2
    metrics = [
        ("top1_accuracy", "Top-1 agreement"),
        ("top3_recall", "Top-3 recall"),
        ("reward_ratio", "Reward ratio"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
    figure.suptitle("Frontier-level quality at a measured open-model cost", x=0.02, ha="left", fontsize=18, fontweight="bold")
    labels = {
        "qwen": "Qwen 9B",
        "gemini": "Gemini",
        "gpt54": "GPT-5.4",
        "gptsol": "GPT-6 Sol",
        "qwen35_2b_sft": "2B + SFT",
        "qwen35_2b_grpo": "2B + GRPO",
    }
    colors = {
        "qwen": ORANGE,
        "gemini": GREEN,
        "gpt54": RED,
        "gptsol": PURPLE,
        "qwen35_2b_sft": BLUE,
        "qwen35_2b_grpo": TEAL,
    }
    offsets = {
        "qwen": (6, 8),
        "gemini": (-44, 8),
        "gpt54": (-48, 8),
        "gptsol": (-62, 8),
        "qwen35_2b_sft": (6, 8),
        "qwen35_2b_grpo": (6, -14),
    }
    for axis, (metric, title) in zip(axes, metrics):
        for model in included:
            key = model["model"]
            if key not in labels:
                continue
            x = float(model["cost_per_1k_recommendations_eur"])
            y = float(model[metric])
            axis.scatter(x, y, color=colors[key], s=65, zorder=3)
            dx, dy = offsets[key]
            axis.annotate(labels[key], (x, y), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=8)
        for model in local:
            key = model["model"]
            y = float(model[metric])
            axis.errorbar(
                midpoint,
                y,
                xerr=np.array([[midpoint - low], [high - midpoint]]),
                fmt="o",
                color=colors[key],
                ecolor=colors[key],
                elinewidth=2,
                capsize=4,
                markersize=7,
                zorder=4,
            )
            dx, dy = offsets[key]
            axis.annotate(labels[key] + "*", (midpoint, y), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=8)
        axis.set_xlim(0, 3.1)
        axis.set_ylim(0.15 if metric == "top1_accuracy" else 0.6, 1.02)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.set_title(title, loc="left")
        axis.set_xlabel("€ per 1,000 recommendations")
        clean_axis(axis, grid_axis="both")
    axes[0].set_ylabel("Metric on 50-scenario golden set")
    add_note(
        figure,
        f"*2B range is measured on one $0.74/h RTX 4090: €{low:.4f}/1k at sustained concurrency 64 to €{high:.4f}/1k without batching. Horizontal axis is linear. "
        f"GPT-6 Sol converted from ${gpt_sol['cost_per_1k_recommendations_usd']:.4f} to €{gpt_sol_eur['cost_per_1k_recommendations_eur']:.4f}/1k at ${cost_assumption['usd_per_eur']:.4f}/€. "
        "Reward-argmax excluded because it is not a serving model.",
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.9))
    return figure


def plot_serving_benchmark(sweep: dict[str, Any], sustained: dict[str, Any]) -> Figure:
    rows = sweep["levels"]
    sustained_rows = sustained["levels"]
    concurrency = [row["concurrency"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    figure.suptitle("Continuous batching turns a small model into cheap capacity", x=0.02, ha="left", fontsize=18, fontweight="bold")

    axis = axes[0]
    axis.plot(concurrency, [row["requests_per_second"] for row in rows], color=BLUE, marker="o", linewidth=2.2)
    axis.scatter(
        [row["concurrency"] for row in sustained_rows],
        [row["requests_per_second"] for row in sustained_rows],
        color=TEAL,
        marker="s",
        s=45,
        label="Sustained 1,200-request run",
        zorder=3,
    )
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("Recommendations / second")
    axis.set_title("Throughput scales to 63.7 rec/s", loc="left")
    clean_axis(axis)
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    axis = axes[1]
    axis.plot(concurrency, [row["latency_seconds"]["p50"] for row in rows], color=GREEN, marker="o", linewidth=2.2, label="p50")
    axis.plot(concurrency, [row["latency_seconds"]["p95"] for row in rows], color=ORANGE, marker="o", linewidth=2.2, label="p95")
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("End-to-end latency (seconds)")
    axis.set_title("Latency remains near one second", loc="left")
    clean_axis(axis)
    axis.legend(frameon=False)

    axis = axes[2]
    axis.plot(concurrency, [row["saturated_cost_eur_per_1k"] for row in rows], color=PURPLE, marker="o", linewidth=2.2)
    axis.scatter(
        [row["concurrency"] for row in sustained_rows],
        [row["saturated_cost_eur_per_1k"] for row in sustained_rows],
        color=TEAL,
        marker="s",
        s=45,
        zorder=3,
    )
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("€ per 1,000 recommendations")
    axis.set_title("Batching cuts GPU cost by 19×", loc="left")
    axis.set_ylim(bottom=0)
    clean_axis(axis)
    best = sustained["best_throughput"]
    axis.annotate(
        f"€{best['saturated_cost_eur_per_1k']:.4f}/1k\n{best['telemetry']['gpu_utilization_pct']['mean']:.1f}% GPU util.",
        (best["concurrency"], best["saturated_cost_eur_per_1k"]),
        xytext=(-92, 34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": PURPLE},
        color=INK,
    )
    add_note(
        figure,
        "Measured with the selected Qwen3.5-2B SFT adapter, vLLM 0.29.0, BF16, and ~1,065 tokens/recommendation on a RunPod RTX 4090 ($0.74/h). "
        "Squares are sustained 1,200-request confirmations; all requests produced valid top-3 JSON.",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.91))
    return figure


def plot_output_efficiency(repo: Path) -> Figure:
    base = metric_summary(repo / "artifacts/training/evaluations/golden-base-qwen35-2b-700tok/summary.json")
    sft = metric_summary(repo / "artifacts/training/evaluations/golden-sft-qwen35-2b-seed3407-700tok/summary.json")
    grpo = metric_summary(repo / "artifacts/training/evaluations/golden-grpo-qwen35-2b-10step-700tok/summary.json")
    rows = [base, sft, grpo]
    stages = ["2B base", "+ SFT", "+ GRPO"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    figure.suptitle("Fine-tuning teaches the model to stop—and to speak JSON", x=0.02, ha="left", fontsize=18, fontweight="bold")

    axis = axes[0]
    bars = axis.bar(stages, [row["mean_completion_tokens"] for row in rows], color=["#AAB5BF", BLUE, TEAL])
    axis.set_ylabel("Mean completion tokens")
    axis.set_title("11× shorter completions", loc="left")
    clean_axis(axis)
    axis.bar_label(bars, labels=[f"{row['mean_completion_tokens']:.0f}" for row in rows], padding=3, color=INK, fontweight="bold")

    axis = axes[1]
    values = [row["strict_json_rate"] for row in rows]
    bars = axis.bar(stages, values, color=["#AAB5BF", BLUE, TEAL])
    axis.set_ylim(0, 1.1)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_ylabel("Strict JSON rate")
    axis.set_title("2% → 100% format compliance", loc="left")
    clean_axis(axis)
    axis.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3, color=INK, fontweight="bold")
    add_note(figure, "Source: frozen evaluations on the same 50-scenario golden set. Throughput is omitted because the SFT and GRPO timing runs were not controlled comparisons.")
    figure.tight_layout(rect=(0, 0.07, 1, 0.91))
    return figure


def source_files(repo: Path) -> list[Path]:
    return [
        repo / "artifacts/quoted/final_distribution_summary.json",
        repo / "artifacts/splits/split_manifest.json",
        repo / "artifacts/teacher/annotation_summary.json",
        repo / "artifacts/reward_calibration/calibration_summary.json",
        repo / "artifacts/training/runs/sft-qwen35-2b-seed3407/run_record.json",
        repo / "artifacts/training/runs/grpo-qwen35-2b-10step/run_record.json",
        repo / "artifacts/training/evaluations/golden-base-qwen35-2b-700tok/summary.json",
        repo / "artifacts/training/evaluations/golden-sft-qwen35-2b-seed3407-700tok/summary.json",
        repo / "artifacts/training/evaluations/golden-grpo-qwen35-2b-10step-700tok/summary.json",
        repo / "artifacts/serving_benchmark/qwen35-2b-sft-vllm-rtx4090-20260926/summary.json",
        repo / "artifacts/serving_benchmark/qwen35-2b-sft-vllm-rtx4090-sustained-20260926/summary.json",
        repo / "results/summary.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = (repo / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    setup_style()

    quoted = read_json(repo / "artifacts/quoted/final_distribution_summary.json")
    splits = read_json(repo / "artifacts/splits/split_manifest.json")
    teacher = read_json(repo / "artifacts/teacher/annotation_summary.json")
    calibration = read_json(repo / "artifacts/reward_calibration/calibration_summary.json")
    sft_record = read_json(repo / "artifacts/training/runs/sft-qwen35-2b-seed3407/run_record.json")
    grpo_record = read_json(repo / "artifacts/training/runs/grpo-qwen35-2b-10step/run_record.json")
    summary = read_json(repo / "results/summary.json")
    sweep = read_json(repo / "artifacts/serving_benchmark/qwen35-2b-sft-vllm-rtx4090-20260926/summary.json")
    sustained = read_json(repo / "artifacts/serving_benchmark/qwen35-2b-sft-vllm-rtx4090-sustained-20260926/summary.json")
    cost_assumption = serving_cost_assumption(sweep, sustained)

    figures: list[tuple[str, Figure]] = [
        ("01-dataset-funnel-and-splits", plot_dataset_funnel_and_splits(quoted, splits, teacher)),
        ("02-dataset-distribution", plot_dataset_distribution(splits, quoted)),
        ("03-reward-calibration", plot_reward_calibration(calibration)),
        ("04-sft-training", plot_sft_training(sft_record)),
        ("05-grpo-training", plot_grpo_training(grpo_record)),
        ("06-model-progress", plot_model_progress(repo, calibration)),
        ("07-cost-quality", plot_cost_quality(summary, cost_assumption)),
        ("08-output-efficiency", plot_output_efficiency(repo)),
        ("09-serving-efficiency", plot_serving_benchmark(sweep, sustained)),
    ]
    outputs: list[str] = []
    for stem, figure in figures:
        outputs.extend(save_figure(figure, output_dir, stem))

    sources = source_files(repo)
    resolved = {
        "seed": args.seed,
        "output_dir": str(output_dir),
        "outputs": outputs,
        "serving_cost_assumption": cost_assumption,
        "sources": [{"path": str(path.relative_to(repo)), "sha256": sha256(path)} for path in sources],
    }
    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(config_path)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
