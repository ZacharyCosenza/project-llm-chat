import torch
from core.utils import print0


def run_sentence_completion(model, tokenizer, device="cuda", max_new_tokens=50, temperature=0.8, top_k=40):
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a world where",
        "The meaning of life is",
        "Hello, my name is",
        "The most important thing to remember is",
        "When I was young",
        "The future of technology",
    ]

    print0("\n" + "="*80)
    print0("SENTENCE COMPLETION VALIDATION")
    print0("="*80)

    model.eval()

    for i, prompt in enumerate(prompts):
        encoded = tokenizer.encode(prompt, return_tensors="pt").to(device)
        completed = model.predict(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        completion_text = tokenizer.decode(completed[0], skip_special_tokens=True)

        print0(f"\n[{i+1}/{len(prompts)}] Prompt: '{prompt}'")
        print0(f"Completion: {completion_text}")
        print0("-" * 80)
    print0("="*80 + "\n")

    model.train()
