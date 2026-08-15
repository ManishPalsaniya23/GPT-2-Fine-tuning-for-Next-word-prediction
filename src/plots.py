"""Training-curve and comparison plots for finished runs.

Usage:
    python -m src.plots                      # every run under outputs/
    python -m src.plots --run gpt2-15ep      # one run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import OUTPUT_ROOT  # noqa: E402

# Categorical slots 1-3 of the reference palette, in fixed order. These three
# validate on the all-pairs list in both modes; a 4th series would need the
# documented fold-to-Other treatment rather than a new hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def short_name(run_name: str) -> str:
    """'gpt2-medium-wikitext-2-raw-v1-lora' -> 'gpt2-medium (lora)'."""
    for marker in ("-wikitext-2-raw-v1-", "-wikitext-"):
        if marker in run_name:
            model, _, mode = run_name.partition(marker)
            return f"{model} ({mode})" if mode else model
    return run_name


def legend_below(ax, ncol: int) -> None:
    """Legend under the plot area - it can never collide with a data label there."""
    legend = ax.legend(
        frameon=False, fontsize=9, loc="upper center",
        bbox_to_anchor=(0.5, -0.16), ncol=ncol, handlelength=1.6,
    )
    for text in legend.get_texts():
        text.set_color(INK)


def style_axes(ax) -> None:
    """Recessive chrome: hairline grid, muted ticks, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def latest_state(run_dir: Path) -> dict | None:
    """trainer_state.json from the highest-numbered checkpoint in a run."""
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for ckpt in reversed(checkpoints):
        state_path = ckpt / "trainer_state.json"
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8"))
    return None


def parse_history(state: dict) -> dict[str, list]:
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    top1, top5, top10 = [], [], []

    for entry in state.get("log_history", []):
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
            top1.append(entry.get("eval_top1_accuracy"))
            top5.append(entry.get("eval_top5_accuracy"))
            top10.append(entry.get("eval_top10_accuracy"))
        elif "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "top1": top1,
        "top5": top5,
        "top10": top10,
    }


def plot_loss(history: dict, run_name: str, out_path: Path) -> None:
    """Train vs validation loss. Divergence between the two is the overfit signal."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axes(ax)

    ax.plot(
        history["train_steps"], history["train_loss"],
        color=SERIES[0], linewidth=2, label="Training loss", zorder=3,
    )
    ax.plot(
        history["eval_steps"], history["eval_loss"],
        color=SERIES[1], linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.5,
        label="Validation loss", zorder=4,
    )

    if history["eval_loss"]:
        best_idx = min(range(len(history["eval_loss"])), key=lambda i: history["eval_loss"][i])
        best_step = history["eval_steps"][best_idx]
        best_val = history["eval_loss"][best_idx]
        ax.axvline(best_step, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.annotate(
            f"best {best_val:.4f}\nstep {best_step}",
            xy=(best_step, best_val), xytext=(8, 14), textcoords="offset points",
            fontsize=9, color=INK,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{short_name(run_name)} - loss", color=INK, fontsize=12, loc="left", pad=12)
    legend_below(ax, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy(history: dict, run_name: str, out_path: Path) -> None:
    """Top-k accuracy. Separate figure from loss - never a second y-axis."""
    if not history["eval_steps"]:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axes(ax)

    for values, color, label in (
        (history["top10"], SERIES[2], "Top-10"),
        (history["top5"], SERIES[1], "Top-5"),
        (history["top1"], SERIES[0], "Top-1"),
    ):
        if not any(v is not None for v in values):
            continue
        pct = [v * 100 if v is not None else None for v in values]
        ax.plot(
            history["eval_steps"], pct, color=color, linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label, zorder=3,
        )
        # Direct label at the line end satisfies the relief rule for the
        # lower-contrast slots, so identity never rests on color alone.
        if pct and pct[-1] is not None:
            ax.annotate(
                f" {label} {pct[-1]:.1f}%",
                xy=(history["eval_steps"][-1], pct[-1]),
                xytext=(6, -3), textcoords="offset points",
                fontsize=9, color=INK, va="center",
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"{short_name(run_name)} - next-token accuracy",
        color=INK, fontsize=12, loc="left", pad=12,
    )
    ax.set_xlim(right=max(history["eval_steps"]) * 1.26)
    legend_below(ax, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_comparison(runs: list[tuple[str, dict]], out_path: Path) -> None:
    """Validation loss across runs on one axis - comparable because the metric is shared."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axes(ax)

    for (name, history), color in zip(runs, SERIES):
        if not history["eval_steps"]:
            continue
        ax.plot(
            history["eval_steps"], history["eval_loss"], color=color, linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=short_name(name), zorder=3,
        )
        ax.annotate(
            f" {short_name(name)} {history['eval_loss'][-1]:.3f}",
            xy=(history["eval_steps"][-1], history["eval_loss"][-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=INK, va="center",
        )

    right = max(max(h["eval_steps"]) for _, h in runs if h["eval_steps"])
    ax.set_xlim(right=right * 1.45)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss by model", color=INK, fontsize=12, loc="left", pad=12)
    legend_below(ax, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves for finished runs")
    parser.add_argument("--run", default=None, help="plot only this run directory name")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    plot_dir = Path(args.plot_dir) if args.plot_dir else root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "plots":
            continue
        if args.run and run_dir.name != args.run:
            continue
        state = latest_state(run_dir)
        if state is None:
            continue
        history = parse_history(state)
        collected.append((run_dir.name, history))

        plot_loss(history, run_dir.name, plot_dir / f"{run_dir.name}_loss.png")
        plot_accuracy(history, run_dir.name, plot_dir / f"{run_dir.name}_accuracy.png")
        print(f"  plotted {run_dir.name} ({len(history['eval_steps'])} evals)")

    if len(collected) > 1:
        plot_comparison(collected[:3], plot_dir / "comparison_loss.png")
        print("  plotted comparison_loss.png")

    if not collected:
        print("No runs with checkpoints found under outputs/.")
    else:
        print(f"\nPlots written to {plot_dir}")


if __name__ == "__main__":
    main()
