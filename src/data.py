"""WikiText loading, tokenization, and packing into fixed-length CLM blocks."""

from __future__ import annotations

from itertools import chain

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import CACHE_ROOT, TrainingConfig

# The canonical wikitext repo moved under the Salesforce org; the bare name is
# still resolvable on older hub clients, so try both.
_DATASET_ALIASES = {"wikitext": ["Salesforce/wikitext", "wikitext"]}


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 ships without a pad token; reusing EOS is safe because the data
    # collator masks padded positions out of the loss.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_raw_dataset(config: TrainingConfig) -> DatasetDict:
    candidates = _DATASET_ALIASES.get(config.dataset_name, [config.dataset_name])
    last_error: Exception | None = None
    for name in candidates:
        try:
            return load_dataset(name, config.dataset_config, cache_dir=str(CACHE_ROOT / "datasets"))
        except Exception as exc:  # noqa: BLE001 - retry the next alias
            last_error = exc
    raise RuntimeError(
        f"Could not load dataset {config.dataset_name}/{config.dataset_config}. "
        f"Last error: {last_error}"
    )


def _group_into_blocks(examples: dict, block_size: int) -> dict:
    """Concatenate tokenized documents, then slice into equal-length blocks.

    Packing this way beats per-line truncation on WikiText: individual lines are
    often only a few tokens, so padding each one would waste most of the batch.
    """
    concatenated = {k: list(chain(*examples[k])) for k in examples}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [v[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, v in concatenated.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result


def build_datasets(
    config: TrainingConfig, tokenizer: PreTrainedTokenizerBase
) -> DatasetDict:
    raw = load_raw_dataset(config)
    text_column = "text" if "text" in raw["train"].column_names else raw["train"].column_names[0]
    num_proc = max(config.dataloader_num_workers, 1)

    tokenized = raw.map(
        lambda batch: tokenizer(batch[text_column]),
        batched=True,
        num_proc=num_proc,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    lm_datasets = tokenized.map(
        lambda batch: _group_into_blocks(batch, config.block_size),
        batched=True,
        num_proc=num_proc,
        desc=f"Packing into {config.block_size}-token blocks",
    )

    if config.max_train_samples is not None:
        lm_datasets["train"] = lm_datasets["train"].select(
            range(min(config.max_train_samples, len(lm_datasets["train"])))
        )
    if config.max_eval_samples is not None:
        for split in ("validation", "test"):
            if split in lm_datasets:
                lm_datasets[split] = lm_datasets[split].select(
                    range(min(config.max_eval_samples, len(lm_datasets[split])))
                )

    return lm_datasets


def describe_datasets(datasets: DatasetDict, block_size: int) -> str:
    lines = ["Dataset splits:"]
    for split, ds in datasets.items():
        tokens = len(ds) * block_size
        lines.append(f"  {split:<12} {len(ds):>8,} blocks  ({tokens:>12,} tokens)")
    return "\n".join(lines)
