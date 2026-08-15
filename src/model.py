"""Model construction: GPT-2 loading, optional LoRA adapters, checkpoint reload."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from .config import TrainingConfig


def _parameter_summary(model: torch.nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    return f"trainable {trainable:,} / total {total:,} parameters ({pct:.2f}%)"


def build_model(
    config: TrainingConfig, tokenizer: PreTrainedTokenizerBase
) -> tuple[PreTrainedModel, str]:
    """Load the base model and, when configured, wrap it in LoRA adapters."""
    # With LoRA the base weights are frozen, so holding them in fp16 halves the
    # resident footprint without affecting gradient quality.
    dtype = torch.float16 if (config.use_lora and (config.fp16 or config.bf16)) else torch.float32
    if config.use_lora and config.bf16:
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # Adapter weights and norms stay in fp32 so the optimizer step is stable
        # even though the frozen base is half precision.
        for name, param in model.named_parameters():
            if param.requires_grad or "ln" in name or "norm" in name.lower():
                param.data = param.data.float()

    if config.gradient_checkpointing:
        model.config.use_cache = False

    return model, _parameter_summary(model)


def load_trained_model(
    model_dir: str | Path, device: torch.device | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Reload a saved run, transparently handling LoRA adapter directories."""
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (model_dir / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, dtype=torch.float32
        )
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, model_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    if device is not None:
        model.to(device)
    return model, tokenizer
