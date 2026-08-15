"""Gradio demo for the next-word predictor.

Usage:
    python app.py --model-dir outputs/gpt2-wikitext-2-raw-v1-full/best_model
"""

from __future__ import annotations

import argparse

import gradio as gr

from src.device import detect_device
from src.model import load_trained_model
from src.predict import predict_next_words


def build_interface(model, tokenizer, device):
    def predict(prompt: str, top_k: int, max_new_tokens: int, temperature: float):
        if not prompt.strip():
            return {}, ""

        candidates = predict_next_words(model, tokenizer, prompt, int(top_k), device)
        distribution = {token: prob for token, prob in candidates}

        import torch

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )
        return distribution, tokenizer.decode(output[0], skip_special_tokens=True)

    with gr.Blocks(title="Next Word Predictor") as demo:
        gr.Markdown("# Next Word Predictor\nFine-tuned GPT-2 trained on WikiText.")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="The history of the Roman Empire",
                    lines=3,
                )
                with gr.Row():
                    top_k = gr.Slider(1, 20, value=10, step=1, label="Candidates to show")
                    max_new = gr.Slider(10, 200, value=50, step=10, label="Continuation length")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
                run = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                label = gr.Label(label="Next word candidates", num_top_classes=20)
        continuation = gr.Textbox(label="Generated continuation", lines=6)

        run.click(predict, [prompt, top_k, max_new, temperature], [label, continuation])
        prompt.submit(predict, [prompt, top_k, max_new, temperature], [label, continuation])

        gr.Examples(
            examples=[
                ["The capital of France is"],
                ["In the early twentieth century, scientists discovered"],
                ["The novel was first published in"],
            ],
            inputs=prompt,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the next-word predictor demo")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    info = detect_device()
    print(f"Loading {args.model_dir} on {info.name}")
    model, tokenizer = load_trained_model(args.model_dir, info.device)

    build_interface(model, tokenizer, info.device).launch(
        server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
