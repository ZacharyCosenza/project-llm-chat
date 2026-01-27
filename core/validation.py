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

    completions = []

    with torch.no_grad():
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

            completions.append({
                "prompt": prompt,
                "completion": completion_text
            })

            del encoded, completed

    print0("="*80 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()

    return completions


def run_world_knowledge_validation(model, tokenizer, device="cuda", max_new_tokens=20, temperature=0.3, top_k=40):
    test_cases = [
        {
            "prompt": "The capital of France is",
            "expected": "The capital of France is Paris",
        },
        {
            "prompt": "The quick brown fox jumps over the",
            "expected": "The quick brown fox jumps over the lazy dog",
        },
        {
            "prompt": "E equals MC",
            "expected": "E equals MC squared",
        },
        {
            "prompt": "To be or not to be, that is the",
            "expected": "To be or not to be, that is the question",
        },
        {
            "prompt": "The first president of the United States was",
            "expected": "The first president of the United States was George Washington",
        },
        {
            "prompt": "Water boils at",
            "expected": "Water boils at 100 degrees Celsius",
        },
        {
            "prompt": "The Earth orbits around the",
            "expected": "The Earth orbits around the Sun",
        },
        {
            "prompt": "Houston, we have a",
            "expected": "Houston, we have a problem",
        },
    ]

    print0("\n" + "="*80)
    print0("WORLD KNOWLEDGE VALIDATION")
    print0("="*80)

    model.eval()

    all_predictions = []
    all_references = []
    all_prompts = []

    with torch.no_grad():
        for i, test_case in enumerate(test_cases):
            prompt = test_case["prompt"]
            expected = test_case["expected"]

            encoded = tokenizer.encode(prompt, return_tensors="pt").to(device)
            completed = model.predict(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            prediction = tokenizer.decode(completed[0], skip_special_tokens=True)

            all_predictions.append(prediction)
            all_references.append(expected)
            all_prompts.append(prompt)

            print0(f"\n[{i+1}/{len(test_cases)}]")
            print0(f"Prompt:    '{prompt}'")
            print0(f"Expected:  '{expected}'")
            print0(f"Generated: '{prediction}'")
            print0("-" * 80)

            del encoded, completed

    # Simple metrics (no external dependencies)
    metrics = {}

    # Exact Match Accuracy
    exact_matches = sum(1 for pred, ref in zip(all_predictions, all_references) if pred.strip() == ref.strip())
    accuracy = exact_matches / len(all_predictions)
    metrics['accuracy'] = float(accuracy)

    # Token-level F1
    total_f1 = 0
    for pred, ref in zip(all_predictions, all_references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            f1 = 1.0
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1 = 0.0
        else:
            intersection = len(pred_tokens & ref_tokens)
            precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = intersection / len(ref_tokens) if len(ref_tokens) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total_f1 += f1

    avg_token_f1 = total_f1 / len(all_predictions)
    metrics['token_f1'] = float(avg_token_f1)

    print0(f"\nExact Match Accuracy: {accuracy:.4f} ({exact_matches}/{len(all_predictions)})")
    print0(f"Token-level F1: {avg_token_f1:.4f}")
    print0("="*80 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()

    return {
        "predictions": all_predictions,
        "references": all_references,
        "prompts": all_prompts,
        "metrics": metrics,
    }
