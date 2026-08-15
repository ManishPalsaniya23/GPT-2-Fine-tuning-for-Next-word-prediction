"""Training configuration, model presets, and VRAM-aware auto-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .device import DeviceInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = PROJECT_ROOT / "cache"

SUPPORTED_MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2")

# Approximate trainable parameter counts, used to predict whether a full
# fine-tune fits in VRAM before we try to allocate anything.
PARAM_COUNTS_MILLIONS = {
    "distilgpt2": 82,
    "gpt2": 124,
    "gpt2-medium": 355,
    "gpt2-large": 774,
    "gpt2-xl": 1558,
}

# Per-model batch/accumulation defaults, sized for a 4 GB laptop GPU. The
# effective batch (batch * accum) is held near 32 so the loss curve stays
# comparable across model sizes.
_BATCH_PRESETS = {
    "distilgpt2": (8, 4),
    "gpt2": (4, 8),
    "gpt2-medium": (2, 16),
    # Batch 2 measured to fit gpt2-large in 4 GB alongside LoRA without
    # checkpointing; it halves the number of forward passes per optimiser step.
    "gpt2-large": (2, 16),
    "gpt2-xl": (1, 32),
}


@dataclass
class TrainingConfig:
    # --- model / data ---
    model_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    block_size: int = 256
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # --- optimisation ---
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    seed: int = 42

    # --- batching ---
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 2

    # --- memory / precision ---
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj"])

    # --- checkpointing / early stopping ---
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # --- metrics ---
    topk_values: list[int] = field(default_factory=lambda: [1, 5, 10])

    # --- io ---
    output_dir: str = ""
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(OUTPUT_ROOT / self.run_name)

    @property
    def run_name(self) -> str:
        tag = "lora" if self.use_lora else "full"
        return f"{self.model_name.replace('/', '-')}-{self.dataset_config}-{tag}"

    @property
    def best_model_dir(self) -> Path:
        return Path(self.output_dir) / "best_model"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimated_full_finetune_gb(model_name: str) -> float:
    """Weights + grads + two Adam moments, all fp32, at 4 bytes each."""
    millions = PARAM_COUNTS_MILLIONS.get(model_name, 124)
    return millions * 1e6 * 16 / (1024**3)


def auto_configure(config: TrainingConfig, info: DeviceInfo) -> tuple[TrainingConfig, list[str]]:
    """Adjust precision, batching, and LoRA to fit the detected hardware.

    Only fields the caller left at their default are touched; anything passed
    explicitly on the command line is preserved. `overrides_applied` carries the
    human-readable list of what changed so train.py can print it.
    """
    notes: list[str] = []
    defaults = TrainingConfig(model_name=config.model_name)
    # Captured before any mutation, so we can tell an explicit --output-dir from
    # the one __post_init__ derived and therefore may safely re-derive.
    output_dir_was_derived = config.output_dir == str(OUTPUT_ROOT / config.run_name)

    def untouched(field_name: str) -> bool:
        return getattr(config, field_name) == getattr(defaults, field_name)

    # --- precision ---
    if info.is_cuda:
        if untouched("bf16") and untouched("fp16"):
            if info.supports_bf16:
                config.bf16 = True
                notes.append("bf16 mixed precision enabled (native support detected)")
            elif info.supports_fp16:
                config.fp16 = True
                notes.append("fp16 mixed precision enabled")
    else:
        config.fp16 = False
        config.bf16 = False
        notes.append("no CUDA device: running on CPU in fp32 (expect this to be slow)")

    # --- LoRA decision ---
    if untouched("use_lora") and info.is_cuda:
        needed = estimated_full_finetune_gb(config.model_name)
        # Leave ~1.2 GB headroom for activations, buffers, and fragmentation.
        if needed > max(info.total_vram_gb - 1.2, 0):
            config.use_lora = True
            notes.append(
                f"LoRA enabled: a full fine-tune of {config.model_name} needs about "
                f"{needed:.1f} GB of optimiser state, over the {info.total_vram_gb:.1f} GB available"
            )

    # --- batching ---
    if untouched("per_device_train_batch_size") and untouched("gradient_accumulation_steps"):
        batch, accum = _BATCH_PRESETS.get(config.model_name, (2, 16))
        if info.is_cuda and info.total_vram_gb >= 12:
            batch, accum = batch * 4, max(accum // 4, 1)
        elif info.is_cuda and info.total_vram_gb >= 8:
            batch, accum = batch * 2, max(accum // 2, 1)
        elif not info.is_cuda:
            batch, accum = 1, 8
        if (batch, accum) != (config.per_device_train_batch_size, config.gradient_accumulation_steps):
            config.per_device_train_batch_size = batch
            config.gradient_accumulation_steps = accum
            notes.append(
                f"batch size {batch} x {accum} accumulation steps "
                f"(effective batch {batch * accum})"
            )
    if untouched("per_device_eval_batch_size"):
        config.per_device_eval_batch_size = max(config.per_device_train_batch_size, 1)

    # --- gradient checkpointing ---
    if untouched("gradient_checkpointing") and info.is_cuda:
        params = PARAM_COUNTS_MILLIONS.get(config.model_name, 124)
        if config.use_lora:
            # LoRA has already removed the optimiser state, so the resident cost
            # is the half-precision base weights plus activations. Only turn on
            # checkpointing when the weights alone claim most of the card --
            # otherwise it buys memory we are not short of and costs an extra
            # recompute forward pass (~30% slower) for nothing.
            resident_gb = params * 1e6 * 2 / (1024**3)
            needs_checkpointing = resident_gb > info.total_vram_gb * 0.45
        else:
            needs_checkpointing = params >= 355 and info.total_vram_gb < 12
        if needs_checkpointing:
            config.gradient_checkpointing = True
            notes.append("gradient checkpointing enabled to trade compute for activation memory")

    # --- LoRA needs a larger LR than a full fine-tune ---
    if config.use_lora and untouched("learning_rate"):
        config.learning_rate = 2e-4
        notes.append("learning rate raised to 2e-4 for LoRA adapters")

    # --- dataloader workers ---
    if untouched("dataloader_num_workers"):
        config.dataloader_num_workers = min(2, max(info.cpu_count - 1, 0))

    if output_dir_was_derived:
        config.output_dir = str(OUTPUT_ROOT / config.run_name)
    return config, notes
