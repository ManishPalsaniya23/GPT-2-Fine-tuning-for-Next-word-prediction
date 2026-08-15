"""Fine-tune a GPT-2 model for next-word prediction on WikiText.

Usage:
    python -m src.train --model gpt2
    python -m src.train --model gpt2-medium --epochs 2
    python -m src.train --model gpt2-large --block-size 256
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import SUPPORTED_MODELS, TrainingConfig, auto_configure
from .data import build_datasets, describe_datasets, load_tokenizer
from .device import describe, detect_device, empty_cache
from .metrics import make_compute_metrics, make_logits_preprocessor, perplexity_from_loss
from .model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 for next-word prediction")
    p.add_argument("--model", default="gpt2", help=f"one of {', '.join(SUPPORTED_MODELS)} or any HF causal LM")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--scheduler", default=None, choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"])
    p.add_argument("--optim", default=None, help="e.g. adamw_torch, adafactor, adamw_torch_fused")
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--patience", type=int, default=None, help="early stopping patience, in evaluations")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--lora", dest="lora", action="store_true", default=None)
    p.add_argument("--no-lora", dest="lora", action="store_false", default=None)
    p.add_argument("--fp16", action="store_true", default=None)
    p.add_argument("--bf16", action="store_true", default=None)
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Build a config, leaving unspecified fields at their defaults so
    auto_configure() knows it is free to tune them."""
    overrides = {
        "model_name": args.model,
        "dataset_name": args.dataset,
        "dataset_config": args.dataset_config,
        "block_size": args.block_size,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "dataloader_num_workers": args.num_workers,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.scheduler,
        "optim": args.optim,
        "eval_steps": args.eval_steps,
        "early_stopping_patience": args.patience,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "resume_from_checkpoint": args.resume,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "use_lora": args.lora,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    config = TrainingConfig(**{k: v for k, v in overrides.items() if v is not None})
    if args.eval_steps is not None:
        config.save_steps = args.eval_steps
    return config


def total_training_steps(config: TrainingConfig, num_examples: int) -> int:
    steps_per_epoch = math.ceil(num_examples / config.effective_batch_size)
    return max(int(steps_per_epoch * config.num_train_epochs), 1)


def build_training_arguments(config: TrainingConfig, num_training_steps: int) -> TrainingArguments:
    # transformers v5 dropped warmup_ratio, so translate it into absolute steps.
    warmup_steps = int(num_training_steps * config.warmup_ratio)

    kwargs = dict(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        # Windows spawns (rather than forks) workers, so without this every
        # evaluation pays the full process start-up cost again.
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        seed=config.seed,
        report_to=[],
        disable_tqdm=False,
    )
    if config.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def build_trainer(config, model, tokenizer, datasets, training_args) -> Trainer:
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_split = "validation" if "validation" in datasets else "test"

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets[eval_split],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(config.topk_values),
        preprocess_logits_for_metrics=make_logits_preprocessor(max(config.topk_values)),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        ],
    )


def save_best_model(trainer: Trainer, tokenizer, config: TrainingConfig) -> Path:
    """Persist the best checkpoint (already reloaded by load_best_model_at_end)."""
    best_dir = config.best_model_dir
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    return best_dir


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    info = detect_device()
    print(describe(info))
    if not info.is_cuda:
        print("  WARNING: CUDA was not detected. Training will fall back to CPU.")
        print("  Reinstall torch with: pip install torch --index-url https://download.pytorch.org/whl/cu126\n")

    config, notes = auto_configure(config, info)
    if notes:
        print("Auto-tuned for this machine:")
        for note in notes:
            print(f"  - {note}")
        print()

    set_seed(config.seed)

    tokenizer = load_tokenizer(config.model_name)
    datasets = build_datasets(config, tokenizer)
    print(describe_datasets(datasets, config.block_size), "\n")

    model, param_summary = build_model(config, tokenizer)
    print(f"Model: {config.model_name} - {param_summary}")
    print(
        f"Effective batch: {config.effective_batch_size} sequences "
        f"x {config.block_size} tokens = {config.effective_batch_size * config.block_size:,} tokens/step\n"
    )

    num_steps = total_training_steps(config, len(datasets["train"]))
    training_args = build_training_arguments(config, num_steps)
    trainer = build_trainer(config, model, tokenizer, datasets, training_args)

    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    metrics = trainer.evaluate()
    metrics["perplexity"] = perplexity_from_loss(metrics["eval_loss"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    best_dir = save_best_model(trainer, tokenizer, config)
    empty_cache()

    print("\n" + "=" * 62)
    print(" Training complete")
    print("=" * 62)
    print(f"  Best model saved to : {best_dir}")
    print(f"  Validation loss     : {metrics['eval_loss']:.4f}")
    print(f"  Perplexity          : {metrics['perplexity']:.2f}")
    for k in config.topk_values:
        key = f"eval_top{k}_accuracy"
        if key in metrics:
            print(f"  {f'Top-{k} accuracy':<20}: {metrics[key] * 100:.2f}%")
    print("=" * 62)
    print(f"\nEvaluate on the test split:\n  python -m src.evaluate --model-dir \"{best_dir}\"")
    print(f"Try it out:\n  python -m src.predict --model-dir \"{best_dir}\" --prompt \"The capital of France is\"")


if __name__ == "__main__":
    main()
