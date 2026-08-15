"""Entry point: verify the GPU setup and print the recommended plan per model.

Run this first:
    python main.py
"""

from __future__ import annotations

import sys

from src.config import (
    PARAM_COUNTS_MILLIONS,
    SUPPORTED_MODELS,
    TrainingConfig,
    auto_configure,
    estimated_full_finetune_gb,
)
from src.device import describe, detect_device


def main() -> int:
    info = detect_device()
    print(describe(info))

    if not info.is_cuda:
        print("\n  CUDA was NOT detected - training would run on CPU and take many hours.")
        print("  Install the CUDA build of PyTorch:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu126\n")
        return 1

    print("\n Recommended plan per model on this GPU\n")
    header = f"  {'model':<14} {'params':>8} {'full-FT VRAM':>13} {'strategy':>10} {'batch x accum':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in SUPPORTED_MODELS:
        config, _ = auto_configure(TrainingConfig(model_name=name), info)
        params = PARAM_COUNTS_MILLIONS[name]
        needed = estimated_full_finetune_gb(name)
        strategy = "LoRA" if config.use_lora else "full"
        batching = f"{config.per_device_train_batch_size} x {config.gradient_accumulation_steps}"
        print(f"  {name:<14} {params:>7}M {needed:>12.1f}G {strategy:>10} {batching:>15}")

    print("\n  'full-FT VRAM' is weights + gradients + Adam moments in fp32. When that")
    print("  exceeds the card, training automatically switches to LoRA adapters.\n")

    print(" Next steps\n")
    print("  1. Train        python -m src.train --model gpt2")
    print("  2. Evaluate     python -m src.evaluate --model-dir outputs/<run>/best_model --compare-baseline")
    print("  3. Predict      python -m src.predict --model-dir outputs/<run>/best_model --interactive")
    print("  4. Demo UI      python app.py --model-dir outputs/<run>/best_model\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
