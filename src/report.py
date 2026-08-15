"""Summarise every finished run in outputs/ as one comparison table.

Usage:
    python -m src.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OUTPUT_ROOT
from .metrics import perplexity_from_loss


def collect_runs(output_root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(output_root.iterdir()):
        best = run_dir / "best_model"
        config_path = best / "training_config.json"
        if not config_path.exists():
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        record = {
            "run": run_dir.name,
            "model": config.get("model_name", "?"),
            "strategy": "LoRA" if config.get("use_lora") else "full",
            "epochs": config.get("num_train_epochs"),
            "lr": config.get("learning_rate"),
            "block": config.get("block_size"),
        }

        # Validation numbers written by train.py.
        eval_path = run_dir / "eval_results.json"
        if eval_path.exists():
            metrics = json.loads(eval_path.read_text(encoding="utf-8"))
            record["val_loss"] = metrics.get("eval_loss")
            record["val_ppl"] = metrics.get("perplexity") or perplexity_from_loss(
                metrics.get("eval_loss", float("nan"))
            )
            for k in (1, 5, 10):
                record[f"val_top{k}"] = metrics.get(f"eval_top{k}_accuracy")

        # Test numbers written by evaluate.py, if it has been run.
        test_path = best / "eval_test.json"
        if test_path.exists():
            results = json.loads(test_path.read_text(encoding="utf-8"))
            tuned = results.get("fine_tuned", {})
            record["test_ppl"] = tuned.get("perplexity")
            record["test_top1"] = tuned.get("top1_accuracy")
            baseline = results.get("baseline")
            if baseline:
                record["base_ppl"] = baseline.get("perplexity")
                record["base_top1"] = baseline.get("top1_accuracy")

        runs.append(record)
    return runs


def _fmt(value, spec: str = ".2f", dash: str = "-") -> str:
    return dash if value is None else format(value, spec)


def print_table(runs: list[dict]) -> None:
    if not runs:
        print("No finished runs found under outputs/. Train a model first.")
        return

    header = (
        f"  {'model':<14} {'mode':<5} {'ep':>3} {'lr':>8} "
        f"{'val ppl':>8} {'top-1':>7} {'top-5':>7} {'top-10':>7} {'test ppl':>9} {'base ppl':>9}"
    )
    print("=" * len(header))
    print(" Run comparison - WikiText-2 next-word prediction")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 4))

    for r in runs:
        print(
            f"  {r['model']:<14} {r['strategy']:<5} {_fmt(r.get('epochs'), '.0f'):>3} "
            f"{_fmt(r.get('lr'), '.0e'):>8} "
            f"{_fmt(r.get('val_ppl')):>8} "
            f"{_fmt((r.get('val_top1') or 0) * 100 if r.get('val_top1') else None):>7} "
            f"{_fmt((r.get('val_top5') or 0) * 100 if r.get('val_top5') else None):>7} "
            f"{_fmt((r.get('val_top10') or 0) * 100 if r.get('val_top10') else None):>7} "
            f"{_fmt(r.get('test_ppl')):>9} {_fmt(r.get('base_ppl')):>9}"
        )

    print("=" * len(header))
    print("  ppl = perplexity (lower is better); accuracies are percentages.")
    print("  'base ppl' is the untuned pretrained model, from --compare-baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all finished training runs")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = parser.parse_args()

    root = Path(args.output_root)
    if not root.exists():
        print(f"No outputs directory at {root}")
        return

    runs = collect_runs(root)
    if args.json:
        print(json.dumps(runs, indent=2))
    else:
        print_table(runs)


if __name__ == "__main__":
    main()
