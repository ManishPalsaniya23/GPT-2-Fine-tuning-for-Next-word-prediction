"""Score a saved checkpoint: perplexity and top-k next-token accuracy.

Usage:
    python -m src.evaluate --model-dir outputs/gpt2-wikitext-2-raw-v1-full/best_model
    python -m src.evaluate --model-dir gpt2 --split test          # untuned baseline
    python -m src.evaluate --model-dir outputs/.../best_model --compare-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from .config import TrainingConfig
from .data import build_datasets, load_tokenizer
from .device import detect_device
from .metrics import evaluate_language_model
from .model import load_trained_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a next-word prediction model")
    p.add_argument("--model-dir", required=True, help="path to a saved run, or a hub id like 'gpt2'")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--block-size", type=int, default=None, help="defaults to the value used in training")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--compare-baseline",
        action="store_true",
        help="also score the untuned pretrained model for a before/after comparison",
    )
    p.add_argument("--output", default=None, help="write results to this JSON file")
    return p.parse_args()


def resolve_block_size(model_dir: str, override: int | None) -> int:
    if override is not None:
        return override
    config_path = Path(model_dir) / "training_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8")).get("block_size", 256)
    return 256


def base_model_name(model_dir: str) -> str:
    config_path = Path(model_dir) / "training_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8")).get("model_name", "gpt2")
    return "gpt2"


def run_evaluation(model, tokenizer, dataset, args, device) -> dict[str, float]:
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return evaluate_language_model(model, loader, device, args.topk)


def print_results(title: str, results: dict[str, float], topk: list[int]) -> None:
    print(f"\n  {title}")
    print(f"    loss          : {results['loss']:.4f}")
    print(f"    perplexity    : {results['perplexity']:.2f}")
    for k in topk:
        print(f"    {f'top-{k} accuracy':<14}: {results[f'top{k}_accuracy'] * 100:.2f}%")
    print(f"    tokens scored : {results['eval_tokens']:,}")


def main() -> None:
    args = parse_args()
    info = detect_device()
    device = info.device
    print(f"Device: {info.name}")

    block_size = resolve_block_size(args.model_dir, args.block_size)
    print(f"Loading model from: {args.model_dir}")
    model, tokenizer = load_trained_model(args.model_dir, device)

    data_config = TrainingConfig(
        model_name=base_model_name(args.model_dir),
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        block_size=block_size,
        dataloader_num_workers=args.num_workers,
    )
    datasets = build_datasets(data_config, tokenizer)
    split = args.split if args.split in datasets else "test"
    dataset = datasets[split]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Split: {split} ({len(dataset):,} blocks of {block_size} tokens)")
    print("\n" + "=" * 62)
    print(" Evaluation results")
    print("=" * 62)

    all_results = {}
    tuned = run_evaluation(model, tokenizer, dataset, args, device)
    all_results["fine_tuned"] = tuned
    print_results("Fine-tuned model", tuned, args.topk)

    if args.compare_baseline:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        baseline_name = base_model_name(args.model_dir)
        print(f"\n  Loading untuned baseline: {baseline_name}")
        baseline_model, _ = load_trained_model(baseline_name, device)
        baseline = run_evaluation(baseline_model, tokenizer, dataset, args, device)
        all_results["baseline"] = baseline
        print_results(f"Baseline ({baseline_name}, untuned)", baseline, args.topk)

        ppl_delta = baseline["perplexity"] - tuned["perplexity"]
        acc_delta = (tuned["top1_accuracy"] - baseline["top1_accuracy"]) * 100
        print("\n  Improvement from fine-tuning")
        print(f"    perplexity    : {ppl_delta:+.2f}  ({'better' if ppl_delta > 0 else 'worse'})")
        print(f"    top-1 accuracy: {acc_delta:+.2f} percentage points")

    print("=" * 62)

    output_path = args.output or str(Path(args.model_dir) / f"eval_{split}.json")
    try:
        Path(output_path).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nResults written to {output_path}")
    except OSError as exc:
        print(f"\nCould not write results to {output_path}: {exc}")


if __name__ == "__main__":
    main()
