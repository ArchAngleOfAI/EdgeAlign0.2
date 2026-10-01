"""Shared pieces for the warm-start ("fill in the removed part") gist pretraining.

Reuse, not reimplementation:
- generator build: diagnostics/_common.load_all -> qwen_dual_embedding.build_softprompt_generator
  (last 3 layers unfrozen there), trainable list: diagnostics/_common.trainable_params
- rollout: distill_softprompt.generate_softprompt math. rollout_gists_serial calls it per example
  (it takes no attention mask); rollout_gists is the batched, left-padded, masked version of the
  same math, checked against the serial one by check (e)
- KD math: the expression from distill_softprompt.main() (kl_div(log_softmax(s/T), softmax(t/T))
  summed over vocab), kept per token here so it can be masked to response tokens only.

New here: the mid-sequence receiver (prefix ++ gist ++ suffix ++ response) and the
sequence construction from separately tokenized pieces.
"""

import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "diagnostics")):
    if p not in sys.path:
        sys.path.insert(0, p)

from distill_softprompt import KD_TEMPERATURE, generate_softprompt  # noqa: E402
from smoke_train import MODEL_PATH  # noqa: E402

DATA_DIR = os.environ.get("WARMSTART_DATA_DIR", "/data/a84460786/warmstart_data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
L = 16
MAX_SEQ = 1024
MIN_REMOVED, MIN_KEPT = 8, 8
MAX_NEW_TOKENS = 256
IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643
PAD_ID = ENDOFTEXT

_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL_PATH)
    return _TOK


def template_parts():
    """(head, tail) of the non-thinking chat template around the user content."""
    tok = get_tokenizer()
    s = tok.apply_chat_template([{"role": "user", "content": "\x00SENTINEL\x00"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    head, tail = s.split("\x00SENTINEL\x00")
    return head, tail


def encode(text):
    return get_tokenizer()(text, add_special_tokens=False)["input_ids"]


def pieces(user_kept, removed_part):
    """Token ids of the pieces, each tokenized separately.

    prefix = template head + user_kept (with its label); removed = raw removed part;
    suffix = rest of the template after the user content (end of user turn, assistant
    header, empty non-thinking block).
    """
    head, tail = template_parts()
    return encode(head + user_kept), encode(removed_part), encode(tail)


# ---------------------------------------------------------------- batching

def build_batch(examples, device):
    """Right-padded teacher / receiver-side token tensors for a list of examples.

    Each example dict needs prefix_ids, removed_ids, suffix_ids, response_ids (response
    includes the closing <|im_end|>) and gap (bool).
    Teacher  = prefix + removed + suffix + response.
    Receiver = prefix + [gist] + suffix + response           (gap)
               prefix + removed + [gist] + suffix + response (no-gap)
    Returns a dict with the teacher ids/mask, the receiver "before-gist" and "after-gist"
    id lists, and for each side the index of the first logit that predicts a response
    token (= response start - 1).
    """
    B = len(examples)
    t_lens = [len(e["prefix_ids"]) + len(e["removed_ids"]) + len(e["suffix_ids"]) + len(e["response_ids"])
              for e in examples]
    T = max(t_lens)
    t_ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    t_mask = torch.zeros((B, T), dtype=torch.long)
    t_start, before, after, r_start, resp_len = [], [], [], [], []
    for i, e in enumerate(examples):
        seq = e["prefix_ids"] + e["removed_ids"] + e["suffix_ids"] + e["response_ids"]
        t_ids[i, :len(seq)] = torch.tensor(seq)
        t_mask[i, :len(seq)] = 1
        t_start.append(len(seq) - len(e["response_ids"]) - 1)
        b = e["prefix_ids"] + ([] if e["gap"] else e["removed_ids"])
        a = e["suffix_ids"] + e["response_ids"]
        before.append(b)
        after.append(a)
        r_start.append(len(b) + L + len(e["suffix_ids"]) - 1)
        resp_len.append(len(e["response_ids"]))
    return {"t_ids": t_ids.to(device), "t_mask": t_mask.to(device), "t_start": t_start,
            "before": before, "after": after, "r_start": r_start, "resp_len": resp_len}


def gather_positions(hidden, starts, lens):
    """Stack hidden[i, starts[i] : starts[i]+lens[i]] into (sum(lens), H)."""
    return torch.cat([hidden[i, s:s + n] for i, (s, n) in enumerate(zip(starts, lens))], dim=0)


@torch.no_grad()
def teacher_response_logits(teacher, batch):
    """Teacher logits at the positions that predict the response tokens, (sum(resp_len), V)."""
    h = teacher.model(input_ids=batch["t_ids"], attention_mask=batch["t_mask"]).last_hidden_state
    return teacher.lm_head(gather_positions(h, batch["t_start"], batch["resp_len"]))


def receiver_inputs(receiver, batch, middle):
    """inputs_embeds = cat(before_embeds, middle, after_embeds), right-padded.

    middle[i] is a (m_i, H) tensor inserted between the before/after pieces
    (the gist, or real embeddings, or an empty tensor for "no gist").
    Returns (inputs_embeds, attention_mask, response start per example).
    """
    emb = receiver.model.embed_tokens
    dev = emb.weight.device
    rows, starts = [], []
    for i, (b, a) in enumerate(zip(batch["before"], batch["after"])):
        m = middle[i]
        parts = [emb(torch.tensor(b, device=dev)), m.to(emb.weight.dtype), emb(torch.tensor(a, device=dev))]
        rows.append(torch.cat(parts, dim=0))
        starts.append(len(b) + m.shape[0] + len(a) - batch["resp_len"][i] - 1)
    T = max(r.shape[0] for r in rows)
    mask = torch.zeros((len(rows), T), dtype=torch.long, device=dev)
    padded = []
    for i, r in enumerate(rows):
        pad = emb(torch.full((T - r.shape[0],), PAD_ID, device=dev))
        padded.append(torch.cat([r, pad], dim=0))
        mask[i, :r.shape[0]] = 1
    embeds = torch.stack(padded, dim=0)
    return embeds, mask, starts


def receiver_response_logits(receiver, batch, middle):
    """NEW receiver for mid-sequence insertion: logits predicting the response tokens."""
    embeds, mask, starts = receiver_inputs(receiver, batch, middle)
    h = receiver.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    return receiver.lm_head(gather_positions(h, starts, batch["resp_len"]))


def kd_per_token(student_logits, teacher_logits):
    """KD math from distill_softprompt.main(), per token (no .mean())."""
    student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)


def split_per_example(values, lens):
    out, i = [], 0
    for n in lens:
        out.append(values[i:i + n])
        i += n
    return out


def per_example_kl(kl_tok, lens):
    """Mean KL over each example's response tokens."""
    return [v.mean().item() for v in split_per_example(kl_tok, lens)]


# ---------------------------------------------------------------- gist rollout with entropy

class LogitCapture:
    """Forward hook on encoder.lm_head: records the logits of every rollout step.

    generate_softprompt calls encoder_model.lm_head once per soft token, so this captures
    the rollout softmax without modifying generate_softprompt.
    """

    def __init__(self, lm_head):
        self.logits = []
        self.active = False
        self.handle = lm_head.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if self.active:
            self.logits.append(out)


def rollout_gists(encoder, embedding2, removed_list, capture, device, pad_id=PAD_ID):
    """Batched rollout: the generate_softprompt math, on a LEFT-padded batch with an attention
    mask and explicit position ids (generate_softprompt itself takes no mask, so it stays
    per-example in rollout_gists_serial). Left padding keeps every example's last real token
    at index -1, so all rows start the soft-token loop together. Same return values as
    rollout_gists_serial; check (e) compares the two."""
    inner = encoder.model
    B, T = len(removed_list), max(len(r) for r in removed_list)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    for i, r in enumerate(removed_list):
        ids[i, T - len(r):] = torch.tensor(r)
        mask[i, T - len(r):] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    capture.logits, capture.active = [], True
    try:
        out = inner(inputs_embeds=inner.embed_tokens(ids), attention_mask=mask, position_ids=pos, use_cache=True)
        past_key_values = out.past_key_values
        hidden = out.last_hidden_state[:, -1:, :]
        next_pos = pos[:, -1:] + 1
        soft_tokens = []
        for step in range(L):
            logits = encoder.lm_head(hidden)
            probs = torch.softmax(logits, dim=-1)
            soft_token = probs @ embedding2.weight
            soft_tokens.append(soft_token)
            if step == L - 1:
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=device)], dim=1)
            out = inner(inputs_embeds=soft_token, attention_mask=mask, position_ids=next_pos,
                        past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            hidden = out.last_hidden_state
            next_pos = next_pos + 1
    finally:
        capture.active = False
    logits = torch.cat(capture.logits, dim=1)  # (B, L, V)
    logp = F.log_softmax(logits, dim=-1)
    return torch.cat(soft_tokens, dim=1), -(logp.exp() * logp).sum(-1), logits.argmax(-1)


def rollout_gists_serial(encoder, embedding2, removed_list, capture, device):
    """One generate_softprompt call per example (no padding). Returns
    (gists (B, L, H), entropy (B, L) differentiable, top1 (B, L) token ids)."""
    gists, ents, tops = [], [], []
    for removed in removed_list:
        ids = torch.tensor([removed], device=device)
        capture.logits, capture.active = [], True
        try:
            g = generate_softprompt(encoder, embedding2, ids, L)
        finally:
            capture.active = False
        logits = torch.cat(capture.logits, dim=1)[0]  # (L, V)
        logp = F.log_softmax(logits, dim=-1)
        ents.append(-(logp.exp() * logp).sum(-1))
        tops.append(logits.argmax(-1))
        gists.append(g[0])
    return torch.stack(gists), torch.stack(ents), torch.stack(tops)
