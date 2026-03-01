import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from core.models import TinyGPT

# --- Preset configuration ---
CHECKPOINT_PATH = '/home/zaccosenza/code/project-llm-chat/logs/w8a44yr1/checkpoints/checkpoint_80069.pt'
N_LAYERS = 20
DIM = N_LAYERS * 64          # 1280
N_HEADS = max(1, (DIM + 127) // 128)  # 10
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 50262
TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = 'You are a helpful assistant.'  # Optional default system prompt (empty = no system turn)

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
def _generate(model, prompt_ids, device, stop_ids):
    idx = prompt_ids.to(device)

    for _ in range(MAX_NEW_TOKENS):
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        logits = model(idx_cond)[:, -1, :] / TEMPERATURE

        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

        if next_id.item() in stop_ids:
            break

    return idx[0, prompt_ids.size(1):].tolist()


def chat():
    device = _get_device()
    print(f'Loading model on {device}...')
    model, tokenizer = _load(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Ready ({n_params:.0f}M params). Type "quit" to exit.\n')

    bos_id  = tokenizer.bos_token_id
    eos_id  = tokenizer.eos_token_id
    usr_id  = tokenizer.convert_tokens_to_ids('<|user|>')
    asst_id = tokenizer.convert_tokens_to_ids('<|assistant|>')
    sys_id  = tokenizer.convert_tokens_to_ids('<|system|>')

    # Stop on EOS (end of conversation) or <|user|> (turn-end signal the model
    # was trained to emit immediately before the next user turn).
    stop_ids = {eos_id, usr_id}

    # Initialise context: BOS [SYS system_prompt]
    # Trained format: BOS [SYS content] USR content ASST content USR content ASST content EOS
    tokens = [bos_id]
    if SYSTEM_PROMPT:
        tokens += [sys_id] + tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)

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

        # Prepend <|user|> only when needed: on the first turn the context ends
        # with BOS (or SYS content), not usr_id. After each assistant turn the
        # context already ends with usr_id (the turn-end signal), so we skip it
        # to avoid a double usr_id.
        if tokens[-1] != usr_id:
            tokens.append(usr_id)
        tokens += tokenizer.encode(user_msg, add_special_tokens=False) + [asst_id]

        generated = _generate(model, torch.tensor([tokens]), device, stop_ids)

        # Strip the stopping token from the displayed response — it's a structural
        # token (<|user|> or EOS), not part of the assistant's answer.
        stop_tok = generated[-1] if generated and generated[-1] in stop_ids else None
        response_toks = generated[:-1] if stop_tok is not None else generated
        response = tokenizer.decode(response_toks, skip_special_tokens=True).strip()
        print(f'Assistant: {response}\n')

        # Extend context: assistant content + usr_id as turn separator.
        # The model emits usr_id when it finishes its turn (trained turn-end signal).
        # If it stopped on EOS or hit MAX_NEW_TOKENS instead, we append usr_id
        # manually so the context stays well-formed for the next user turn.
        tokens += response_toks + [usr_id]


if __name__ == '__main__':
    chat()
