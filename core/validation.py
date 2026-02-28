import os
import csv
import json
import random
import yaml
import torch
import torch.nn.functional as F
from jinja2 import Template
from core.utils import print0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'eval_data')


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
        "It was a dark and stormy",
        "Scientists recently discovered that",
        "The best way to learn is",
        "She opened the door and",
        "In the beginning there was",
        "Every great journey begins with",
        "The sun set over the",
        "If I could change one thing",
        "They say that knowledge is",
        "Deep in the forest there",
        "The key to success is",
        "Long ago in a distant",
        "Music has the power to",
        "At the end of the day",
        "The old man looked at",
        "Breaking news today as",
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
            generated_part = completion_text[len(prompt):]

            print0(f"  {prompt} | {generated_part}")

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
        {
            "prompt": "The largest ocean on Earth is the",
            "expected": "The largest ocean on Earth is the Pacific Ocean",
        },
        {
            "prompt": "I think, therefore I",
            "expected": "I think, therefore I am",
        },
        {
            "prompt": "The chemical symbol for water is",
            "expected": "The chemical symbol for water is H2O",
        },
        {
            "prompt": "One small step for man, one giant leap for",
            "expected": "One small step for man, one giant leap for mankind",
        },
        {
            "prompt": "The theory of relativity was proposed by",
            "expected": "The theory of relativity was proposed by Albert Einstein",
        },
        {
            "prompt": "The tallest mountain in the world is",
            "expected": "The tallest mountain in the world is Mount Everest",
        },
        {
            "prompt": "The human body has 206",
            "expected": "The human body has 206 bones",
        },
        {
            "prompt": "The Mona Lisa was painted by",
            "expected": "The Mona Lisa was painted by Leonardo da Vinci",
        },
        {
            "prompt": "The largest planet in our solar system is",
            "expected": "The largest planet in our solar system is Jupiter",
        },
        {
            "prompt": "Four score and seven years",
            "expected": "Four score and seven years ago",
        },
        {
            "prompt": "Romeo, Romeo, wherefore art thou",
            "expected": "Romeo, Romeo, wherefore art thou Romeo",
        },
        {
            "prompt": "The speed of light is approximately",
            "expected": "The speed of light is approximately 300,000 kilometers per second",
        },
        {
            "prompt": "The Great Wall of China was built to",
            "expected": "The Great Wall of China was built to protect against invasions",
        },
        {
            "prompt": "Shakespeare was born in",
            "expected": "Shakespeare was born in Stratford-upon-Avon",
        },
        {
            "prompt": "Ask not what your country can do for you",
            "expected": "Ask not what your country can do for you ask what you can do for your country",
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

            generated_part = prediction[len(prompt):]
            expected_part = expected[len(prompt):]
            match = "Y" if prediction.strip() == expected.strip() else "N"
            print0(f"  [{match}] {prompt} | {generated_part}  (expected: {expected_part})")

            del encoded, completed

    metrics = {}

    exact_matches = sum(1 for pred, ref in zip(all_predictions, all_references) if pred.strip() == ref.strip())
    accuracy = exact_matches / len(all_predictions)
    metrics['accuracy'] = float(accuracy)

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


def run_conversation_validation(model, tokenizer, device="cuda", max_new_tokens=80, temperature=0.7, top_k=40):
    """Validates model conversation capability using special tokens.

    Covers:
    - Same user input, different system prompts (tests system conditioning)
    - Different user inputs, same system prompt (tests instruction following)
    - Multi-turn exchanges (tests context tracking)
    - Mix of world knowledge and creative prompts
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    user_id = tokenizer.convert_tokens_to_ids('<|user|>')
    asst_id = tokenizer.convert_tokens_to_ids('<|assistant|>')
    sys_id = tokenizer.convert_tokens_to_ids('<|system|>')

    def encode_conversation(messages):
        """Encode messages with special tokens; primes the assistant turn at end."""
        tokens = [bos_id]
        for msg in messages:
            if msg['role'] == 'system':
                tokens.append(sys_id)
            elif msg['role'] == 'user':
                tokens.append(user_id)
            elif msg['role'] == 'assistant':
                tokens.append(asst_id)
            tokens.extend(tokenizer.encode(msg['content'], add_special_tokens=False))
        tokens.append(asst_id)  # prime assistant response
        return tokens

    test_cases = [
        # --- Same user input, different system prompts ---
        {
            "group": "system_variation",
            "desc": "Formal system + capital question",
            "messages": [
                {"role": "system", "content": "You are a formal and concise assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
        {
            "group": "system_variation",
            "desc": "Pirate system + capital question",
            "messages": [
                {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
        {
            "group": "system_variation",
            "desc": "No system + capital question",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
        # --- Different user inputs, same system prompt ---
        {
            "group": "user_variation",
            "desc": "World knowledge: largest planet",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the largest planet in our solar system?"},
            ],
        },
        {
            "group": "user_variation",
            "desc": "World knowledge: speed of light",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the speed of light?"},
            ],
        },
        {
            "group": "user_variation",
            "desc": "Creative: haiku about the ocean",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a haiku about the ocean."},
            ],
        },
        {
            "group": "user_variation",
            "desc": "Creative: short story opener",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Begin a one-sentence story about a lost astronaut."},
            ],
        },
        # --- Multi-turn: world knowledge follow-up ---
        {
            "group": "multiturn",
            "desc": "Shakespeare follow-up",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who wrote Romeo and Juliet?"},
                {"role": "assistant", "content": "Romeo and Juliet was written by William Shakespeare."},
                {"role": "user", "content": "When was he born?"},
            ],
        },
        {
            "group": "multiturn",
            "desc": "Science follow-up",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is photosynthesis?"},
                {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight into food using carbon dioxide and water."},
                {"role": "user", "content": "Which part of the plant does this happen in?"},
            ],
        },
        # --- Multi-turn: creative continuation ---
        {
            "group": "multiturn",
            "desc": "Story continuation",
            "messages": [
                {"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": "Start a story about a dragon."},
                {"role": "assistant", "content": "Once upon a time, a small purple dragon lived alone in a cave beneath a rainbow."},
                {"role": "user", "content": "What happens next?"},
            ],
        },
        # --- Creative: standalone ---
        {
            "group": "creative",
            "desc": "Poem about autumn",
            "messages": [
                {"role": "system", "content": "You are a poet who writes in vivid imagery."},
                {"role": "user", "content": "Write a short poem about autumn leaves."},
            ],
        },
        {
            "group": "creative",
            "desc": "Space station story",
            "messages": [
                {"role": "system", "content": "You are a science fiction writer."},
                {"role": "user", "content": "Describe the first morning on a new space station."},
            ],
        },
    ]

    # Tokens to strip from displayed response
    _special_display = ['<|user|>', '<|assistant|>', '<|system|>', '<|beginoftext|>', '<|pad|>']

    print0("\n" + "=" * 80)
    print0("CONVERSATION VALIDATION")
    print0("=" * 80)

    model.eval()
    completions = []

    with torch.no_grad():
        for tc in test_cases:
            tokens = encode_conversation(tc['messages'])
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

            output_ids = model.predict(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            # Decode only the newly generated tokens
            new_token_ids = output_ids[0, len(tokens):].tolist()
            # Truncate at EOS or <|user|>: EOS ends the full conversation; <|user|>
            # is the turn-end signal the model emits after intermediate assistant turns.
            for stop_id in (eos_id, user_id):
                if stop_id in new_token_ids:
                    new_token_ids = new_token_ids[:new_token_ids.index(stop_id)]

            response = tokenizer.decode(new_token_ids, skip_special_tokens=False)
            for tok in _special_display:
                response = response.replace(tok, '')
            response = response.strip()

            last_user = next((m['content'] for m in reversed(tc['messages']) if m['role'] == 'user'), '')
            system = next((m['content'] for m in tc['messages'] if m['role'] == 'system'), None)

            print0(f"\n  [{tc['group']}] {tc['desc']}")
            if system:
                print0(f"  System: {system[:70]}")
            print0(f"  User:   {last_user[:80]}")
            print0(f"  Model:  {response[:120]}")

            completions.append({
                "group": tc['group'],
                "desc": tc['desc'],
                "system": system or "",
                "user": last_user,
                "response": response,
            })

            del input_ids, output_ids

    print0("=" * 80 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    return completions


# --- CORE Evaluation (DCLM benchmark) ---

def render_prompts_mc(item, delimiter, fewshot):
    tpl = Template(
        "{%- for ex in fewshot -%}{{ ex.query }}{{ d }}{{ ex.choices[ex.gold] }}\n\n"
        "{% endfor -%}{{ item.query }}{{ d }}{{ choice }}"
    )
    return [tpl.render(fewshot=fewshot, d=delimiter, item=item, choice=c) for c in item['choices']]


def render_prompts_schema(item, delimiter, fewshot):
    tpl = Template(
        "{%- for ex in fewshot -%}{{ ex.context_options[ex.gold] }}{{ d }}{{ ex.continuation }}\n\n"
        "{% endfor -%}{{ ctx }}{{ d }}{{ item.continuation }}"
    )
    return [tpl.render(fewshot=fewshot, d=delimiter, item=item, ctx=co) for co in item['context_options']]


def render_prompts_lm(item, delimiter, fewshot):
    base = Template(
        "{%- for ex in fewshot -%}{{ ex.context | trim }}{{ d }}{{ ex.continuation }}\n\n"
        "{% endfor -%}{{ item.context | trim }}{{ d }}"
    )
    full = Template(
        "{%- for ex in fewshot -%}{{ ex.context | trim }}{{ d }}{{ ex.continuation }}\n\n"
        "{% endfor -%}{{ item.context | trim }}{{ d }}{{ item.continuation }}"
    )
    ctx = dict(fewshot=fewshot, d=delimiter, item=item)
    return [base.render(**ctx).strip(), full.render(**ctx)]


def find_common_length(seqs, direction='left'):
    min_len = min(len(s) for s in seqs)
    indices = range(min_len) if direction == 'left' else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        if not all(s[idx] == seqs[0][idx] for s in seqs):
            return i
    return min_len


def tokenize_prompts(tokenizer, prompts):
    return [tokenizer.encode(p, add_special_tokens=False) for p in prompts]


def batch_mc(tokenizer, prompts):
    tokens = tokenize_prompts(tokenizer, prompts)
    prefix_len = find_common_length(tokens, 'left')
    return tokens, [prefix_len] * len(prompts), [len(t) for t in tokens]


def batch_schema(tokenizer, prompts):
    tokens = tokenize_prompts(tokenizer, prompts)
    suffix_len = find_common_length(tokens, 'right')
    ends = [len(t) for t in tokens]
    return tokens, [e - suffix_len for e in ends], ends


def batch_lm(tokenizer, prompts):
    tokens = tokenize_prompts(tokenizer, prompts)
    t_without, t_with = tokens
    assert t_without == t_with[:len(t_without)]
    return [t_with], [len(t_without)], [len(t_with)]


def stack_pad(tokens, pad_id):
    bsz = len(tokens)
    seq_len = max(len(t) for t in tokens)
    ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens):
        ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return ids


@torch.no_grad()
def forward_get_losses(model, input_ids, pad_id=None):
    padding_mask = (input_ids == pad_id) if pad_id is not None else None
    logits = model(input_ids, padding_mask=padding_mask)
    B, T, V = logits.shape
    targets = torch.roll(input_ids, -1, 1)
    losses = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), reduction='none').view(B, T)
    losses[:, -1] = float('nan')
    preds = logits.argmax(dim=-1)
    return losses, preds


@torch.no_grad()
def evaluate_example(model, tokenizer, item, data, idx, task_meta, device, max_seq_len):
    task_type = task_meta['task_type']
    delimiter = task_meta['continuation_delimiter']
    num_fewshot = task_meta['num_fewshot']

    fewshot = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        fewshot = [data[i] for i in rng.sample(available, min(num_fewshot, len(available)))]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, delimiter, fewshot)
        tokens, starts, ends = batch_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, delimiter, fewshot)
        tokens, starts, ends = batch_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, delimiter, fewshot)
        tokens, starts, ends = batch_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if max_seq_len:
        new_tokens, new_starts, new_ends = [], [], []
        for t, s, e in zip(tokens, starts, ends):
            if len(t) > max_seq_len:
                crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_starts.append(s - crop)
                new_ends.append(e - crop)
            else:
                new_tokens.append(t)
                new_starts.append(s)
                new_ends.append(e)
        tokens, starts, ends = new_tokens, new_starts, new_ends

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = stack_pad(tokens, pad_id).to(device)
    use_pad_mask = tokenizer.pad_token_id is not None
    losses, preds = forward_get_losses(model, input_ids, pad_id=pad_id if use_pad_mask else None)

    if task_type == 'language_modeling':
        si, ei = starts[0], ends[0]
        predicted = preds[0, si - 1:ei - 1]
        actual = input_ids[0, si:ei]
        is_correct = torch.all(predicted == actual).item()
        pred_text = tokenizer.decode(predicted.tolist())
        gold_text = tokenizer.decode(actual.tolist())
    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, s - 1:e - 1].mean().item() for i, (s, e) in enumerate(zip(starts, ends))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
        if task_type == 'multiple_choice':
            pred_text = item['choices'][pred_idx]
            gold_text = item['choices'][item['gold']]
        else:
            pred_text = item['context_options'][pred_idx]
            gold_text = item['context_options'][item['gold']]

    return is_correct, pred_text, gold_text


def load_random_baselines(eval_dir):
    meta_path = os.path.join(eval_dir, "eval_meta_data.csv")
    baselines = {}
    with open(meta_path, 'r') as f:
        for row in csv.DictReader(f):
            baselines[row['Eval Task']] = float(row['Random baseline'])
    return baselines


def run_core_eval(model, tokenizer, device, num_examples=27):
    eval_dir = os.path.abspath(EVAL_DIR)
    if not os.path.exists(eval_dir):
        print0("CORE eval data not found. Run: python -m core.dataset --eval")
        return None

    config_path = os.path.join(eval_dir, "core.yaml")
    data_dir = os.path.join(eval_dir, "eval_data")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    random_baselines = load_random_baselines(eval_dir)

    tasks_by_type = {}
    for t in config['icl_tasks']:
        tasks_by_type.setdefault(t['icl_task_type'], []).append(t)

    per_type = num_examples // 3
    remainder = num_examples % 3
    type_order = list(tasks_by_type.keys())

    print0("\n" + "=" * 80)
    print0("CORE EVALUATION")
    print0("=" * 80)

    model.eval()
    max_seq_len = model.pos_emb.num_embeddings

    task_results = {}
    # Track how many examples have been printed per task label (capped at 3)
    prints_per_label = {}

    for type_idx, task_type in enumerate(type_order):
        n = per_type + (1 if type_idx < remainder else 0)
        if n == 0:
            continue

        type_tasks = tasks_by_type[task_type]
        rng = random.Random(42)
        rng.shuffle(type_tasks)

        print0(f"\n--- {task_type} ({n} examples) ---")

        task_data = {}
        task_cursors = {}
        for task in type_tasks:
            label = task['label']
            data_path = os.path.join(data_dir, task['dataset_uri'])
            with open(data_path) as f:
                data = [json.loads(line) for line in f]
            rng_data = random.Random(1337)
            rng_data.shuffle(data)
            task_data[label] = data
            task_cursors[label] = 0

        examples_done = 0
        while examples_done < n:
            made_progress = False
            for task in type_tasks:
                if examples_done >= n:
                    break

                label = task['label']
                cursor = task_cursors[label]
                data = task_data[label]
                if cursor >= len(data):
                    continue

                task_meta = {
                    'task_type': task['icl_task_type'],
                    'num_fewshot': task['num_fewshot'][0],
                    'continuation_delimiter': task.get('continuation_delimiter', ' ')
                }

                try:
                    correct, pred, gold = evaluate_example(
                        model, tokenizer, data[cursor], data, cursor, task_meta, device, max_seq_len
                    )
                except Exception as e:
                    print0(f"  [{label}] Example {cursor}: ERROR - {e}")
                    task_cursors[label] = cursor + 1
                    made_progress = True
                    continue

                task_cursors[label] = cursor + 1
                made_progress = True

                mark = "Y" if correct else "N"
                if label not in task_results:
                    task_results[label] = [0, 0]
                task_results[label][0] += int(correct)
                task_results[label][1] += 1
                examples_done += 1

                # Print at most 3 examples per task label
                if prints_per_label.get(label, 0) < 3:
                    pred_short = pred[:80].replace('\n', ' ')
                    gold_short = gold[:80].replace('\n', ' ')
                    print0(f"  [{mark}] {label}: pred={pred_short!r}  gold={gold_short!r}")
                    prints_per_label[label] = prints_per_label.get(label, 0) + 1

            if not made_progress:
                break

    raw_accuracies = {}
    centered_scores = {}
    for label, (correct, total) in task_results.items():
        acc = correct / total if total > 0 else 0.0
        raw_accuracies[label] = acc

        baseline = random_baselines.get(label, 0.0)
        baseline_frac = baseline / 100.0
        if baseline_frac < 1.0:
            centered = (acc - baseline_frac) / (1.0 - baseline_frac)
        else:
            centered = 0.0
        centered_scores[label] = centered

    core_metric = sum(centered_scores.values()) / len(centered_scores) if centered_scores else 0.0

    total_correct = sum(c for c, _ in task_results.values())
    total_evaluated = sum(t for _, t in task_results.values())
    raw_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    print0(f"\n{'Task':<35} {'Acc':>8} {'Baseline':>10} {'Centered':>10}")
    print0("-" * 65)
    for label in task_results:
        acc = raw_accuracies[label]
        baseline = random_baselines.get(label, 0.0)
        centered = centered_scores[label]
        print0(f"  {label:<33} {acc:>8.4f} {baseline:>9.1f}% {centered:>10.4f}")

    print0("-" * 65)
    print0(f"  Raw Accuracy: {total_correct}/{total_evaluated} = {raw_accuracy:.4f}")
    print0(f"  CORE Metric (mean centered): {core_metric:.4f}")
    print0("=" * 80 + "\n")

    return {
        "correct": total_correct,
        "total": total_evaluated,
        "raw_accuracy": raw_accuracy,
        "core_metric": core_metric,
        "per_task_accuracy": raw_accuracies,
        "per_task_centered": centered_scores,
    }
