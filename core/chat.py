import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from core.models import TinyGPT

# --- Preset configuration ---
CHECKPOINT_PATH = '/home/zaccosenza/code/project-llm-chat/logs/771w9gdd/checkpoints/checkpoint_70000.pt'
N_LAYERS = 20
DIM = N_LAYERS * 64          # 1280
N_HEADS = max(1, (DIM + 127) // 128)  # 10
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 50262
TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 300


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')


def _load(device):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({
        'bos_token': '<|beginoftext|>',
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|system|>'],
    })

    model = TinyGPT(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # Resize embeddings to cover the special tokens added above
    old_vocab = model.tok_emb.num_embeddings
    new_vocab = len(tokenizer)
    if old_vocab < new_vocab:
        old_tok = model.tok_emb.weight.data.clone()
        old_head = model.head.weight.data.clone()
        model.tok_emb = nn.Embedding(new_vocab, DIM)
        model.head = nn.Linear(DIM, new_vocab, bias=False)
        model.tok_emb.weight.data[:old_vocab] = old_tok
        model.head.weight.data[:old_vocab] = old_head

    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _generate(model, tokenizer, prompt_ids, device):
    idx = prompt_ids.to(device)
    eos_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        logits = model(idx_cond)[:, -1, :] / TEMPERATURE

        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_id:
            break

        idx = torch.cat((idx, next_id), dim=1)

    return idx[0, prompt_ids.size(1):].tolist()


def chat():
    device = _get_device()
    print(f'Loading model on {device}...')
    model, tokenizer = _load(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Ready ({n_params:.0f}M params). Type "quit" to exit.\n')

    tokens = [tokenizer.bos_token_id]
    user_tok = tokenizer.convert_tokens_to_ids('<|user|>')
    asst_tok = tokenizer.convert_tokens_to_ids('<|assistant|>')

    while True:
        try:
            user_msg = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_msg.lower() in ('quit', 'exit', 'q'):
            break
        if not user_msg:
            continue

        tokens.append(user_tok)
        tokens.extend(tokenizer.encode(user_msg, add_special_tokens=False))
        tokens.append(asst_tok)

        prompt_ids = torch.tensor([tokens])
        generated = _generate(model, tokenizer, prompt_ids, device)

        response = tokenizer.decode(generated).strip()
        print(f'Assistant: {response}\n')

        tokens.extend(generated)
        tokens.append(tokenizer.eos_token_id)


if __name__ == '__main__':
    chat()
