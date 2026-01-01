import torch
from core.utils import print0
import evaluate
import numpy as np

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

    with torch.no_grad():  # Prevent gradient computation during validation
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

            # Clean up GPU tensors immediately
            del encoded, completed

    print0("="*80 + "\n")

    # Clear GPU cache after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()

    return completions


def run_world_knowledge_validation(model, tokenizer, device="cuda", max_new_tokens=20, temperature=0.3, top_k=40):

    # Predictable sentence prompts with their expected completions
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

    # Load evaluation metrics
    print0("\n" + "="*80)
    print0("LOADING EVALUATION METRICS")
    print0("="*80)

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bertscore_metric = evaluate.load("bertscore")

    print0("\n" + "="*80)
    print0("WORLD KNOWLEDGE VALIDATION WITH METRICS")
    print0("="*80)

    model.eval()

    all_predictions = []
    all_references = []
    all_prompts = []

    # Generate predictions for all test cases
    with torch.no_grad():  # Prevent gradient computation during validation
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

            # Clean up GPU tensors immediately
            del encoded, completed

    # Calculate metrics
    print0("\n" + "="*80)
    print0("AGGREGATE METRICS")
    print0("="*80 + "\n")

    # Initialize metrics dictionary
    metrics = {}

    # BLEU Score
    try:
        bleu_results = bleu_metric.compute(
            predictions=all_predictions,
            references=[[ref] for ref in all_references]
        )
        metrics['bleu_score'] = float(bleu_results['bleu'])
        print0(f"BLEU Score: {bleu_results['bleu']:.4f}")
        print0(f"  - BLEU-1: {bleu_results['precisions'][0]:.4f}")
        print0(f"  - BLEU-2: {bleu_results['precisions'][1]:.4f}")
        print0(f"  - BLEU-3: {bleu_results['precisions'][2]:.4f}")
        print0(f"  - BLEU-4: {bleu_results['precisions'][3]:.4f}")
    except Exception as e:
        print0(f"BLEU calculation failed: {e}")
        metrics['bleu_score'] = 0.0

    # ROUGE Score
    try:
        rouge_results = rouge_metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        metrics['rouge1'] = float(rouge_results['rouge1'])
        metrics['rouge2'] = float(rouge_results['rouge2'])
        metrics['rougeL'] = float(rouge_results['rougeL'])
        print0(f"\nROUGE Scores:")
        print0(f"  - ROUGE-1: {rouge_results['rouge1']:.4f}")
        print0(f"  - ROUGE-2: {rouge_results['rouge2']:.4f}")
        print0(f"  - ROUGE-L: {rouge_results['rougeL']:.4f}")
    except Exception as e:
        print0(f"ROUGE calculation failed: {e}")
        metrics['rouge1'] = 0.0
        metrics['rouge2'] = 0.0
        metrics['rougeL'] = 0.0

    # METEOR Score
    try:
        meteor_results = meteor_metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        metrics['meteor_score'] = float(meteor_results['meteor'])
        print0(f"\nMETEOR Score: {meteor_results['meteor']:.4f}")
    except Exception as e:
        print0(f"METEOR calculation failed: {e}")
        metrics['meteor_score'] = 0.0

    # BERTScore
    try:
        bertscore_results = bertscore_metric.compute(
            predictions=all_predictions,
            references=all_references,
            lang="en"
        )
        avg_precision = np.mean(bertscore_results['precision'])
        avg_recall = np.mean(bertscore_results['recall'])
        avg_f1 = np.mean(bertscore_results['f1'])

        metrics['bertscore_precision'] = float(avg_precision)
        metrics['bertscore_recall'] = float(avg_recall)
        metrics['bertscore_f1'] = float(avg_f1)

        print0(f"\nBERTScore:")
        print0(f"  - Precision: {avg_precision:.4f}")
        print0(f"  - Recall:    {avg_recall:.4f}")
        print0(f"  - F1:        {avg_f1:.4f}")
    except Exception as e:
        print0(f"BERTScore calculation failed: {e}")
        metrics['bertscore_precision'] = 0.0
        metrics['bertscore_recall'] = 0.0
        metrics['bertscore_f1'] = 0.0

    # Exact Match Accuracy
    exact_matches = sum(1 for pred, ref in zip(all_predictions, all_references) if pred.strip() == ref.strip())
    accuracy = exact_matches / len(all_predictions)
    metrics['accuracy'] = float(accuracy)
    print0(f"\nExact Match Accuracy: {accuracy:.4f} ({exact_matches}/{len(all_predictions)})")

    # Token-level F1 (simple implementation)
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
    print0(f"Average Token-level F1: {avg_token_f1:.4f}")

    print0("\n" + "="*80 + "\n")

    # Clean up metric objects to free memory (especially BERTScore model)
    del bleu_metric, rouge_metric, meteor_metric, bertscore_metric

    # Clear GPU cache after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()

    return {
        "predictions": all_predictions,
        "references": all_references,
        "prompts": all_prompts,
        "metrics": metrics,
    }
