"""Next-word prediction and text continuation from a trained checkpoint.

Usage:
    python -m src.predict --model-dir outputs/.../best_model --prompt "The capital of France is"
    python -m src.predict --model-dir outputs/.../best_model --interactive
"""

from __future__ import annotations

import argparse

import torch

from .device import detect_device
from .model import load_trained_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict the next word with a fine-tuned GPT-2")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--top-k", type=int, default=10, help="how many candidate next words to show")
    p.add_argument("--interactive", action="store_true", help="loop reading prompts from stdin")
    p.add_argument("--generate", action="store_true", help="also generate a longer continuation")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--nucleus-p", type=float, default=0.9)
    return p.parse_args()


@torch.no_grad()
def predict_next_words(model, tokenizer, prompt: str, top_k: int, device) -> list[tuple[str, float]]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    return [
        (tokenizer.decode(token_id).replace("\n", "\\n"), prob.item())
        for token_id, prob in zip(top_ids, top_probs)
    ]


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt: str, args, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.nucleus_p,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def show(model, tokenizer, prompt: str, args, device) -> None:
    predictions = predict_next_words(model, tokenizer, prompt, args.top_k, device)
    print(f'\n  Prompt: "{prompt}"')
    print(f"  Top {args.top_k} next-word candidates:")
    for rank, (token, prob) in enumerate(predictions, start=1):
        bar = "#" * max(int(prob * 40), 0)
        print(f"    {rank:>2}. {token!r:<16} {prob * 100:>6.2f}%  {bar}")

    if args.generate:
        print("\n  Generated continuation:")
        print(f"    {generate_continuation(model, tokenizer, prompt, args, device)}")


def main() -> None:
    args = parse_args()
    info = detect_device()
    model, tokenizer = load_trained_model(args.model_dir, info.device)
    print(f"Loaded {args.model_dir} on {info.name}")

    if args.prompt:
        show(model, tokenizer, args.prompt, args, info.device)

    if args.interactive or not args.prompt:
        print("\nInteractive mode - type a prompt, or an empty line to quit.")
        while True:
            try:
                prompt = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            show(model, tokenizer, prompt, args, info.device)
        print("\nBye.")


if __name__ == "__main__":
    main()
