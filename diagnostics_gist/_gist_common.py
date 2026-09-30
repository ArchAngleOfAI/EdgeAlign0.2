"""Shared setup for the gist-health diagnostics (diagnostics_gist/t*.py).

Reuses, never reimplements:
- model build + unfreezing: diagnostics/_common.load_all (-> qwen_dual_embedding.build_softprompt_generator),
  called with num_trainable_layers=3 exactly as train_3layer.py does;
- checkpoint I/O: train_3layer.trainable_state / load_trainable_state;
- held-out prompts: train_3layer.heldout_batches (FineWeb-Edu shard 001_00000, never read by training);
- rollout / receiver / loss: distill_softprompt.generate_softprompt, distill_softprompt.receiver_forward,
  and the verbatim training KD loss (diagnostics/_common.kd_loss).

The gist is always a PREFIX before the prompt (receiver_forward's torch.cat order), L=16, prompt
length 128, fp32, and every KL is taken on the prompt positions after the gist, aligned with the
teacher exactly as in training (receiver_forward slices off the prefix positions).
"""

import json
import os
import sys

import torch

GIST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(GIST_DIR)
RESULTS_DIR = os.path.join(GIST_DIR, "results")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "diagnostics"))

import smoke_train  # noqa: E402
import train_3layer as T  # noqa: E402
from _common import (  # noqa: E402
    MODEL_PATH, generate_softprompt, get_device, get_tokenizer, kd_loss, load_all, receiver_forward,
    setup_stdout,
)

L = T.L                      # 16
NUM_LAYERS = T.NUM_LAYERS    # 3
CKPT_DIR = T.DEFAULT_CKPT_DIR
CKPT_FILES = {"step1000": "step_01000.pt", "best": "best.pt"}
TEXT16 = "You are a careful coding assistant. Read the following text and continue it accurately."
N_EXTRA = 256
SEED = 0


def save_json(name, obj):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"saved {path}")


class Setup:
    """Models loaded once; checkpoints swapped in-place via train_3layer.load_trainable_state."""

    def __init__(self, load_ckpts=True):
        setup_stdout()
        torch.manual_seed(SEED)
        self.device = get_device()
        self.encoder, self.embedding2, _, self.teacher, self.receiver = load_all(
            self.device, num_trainable_layers=NUM_LAYERS)
        self.encoder.config.tie_word_embeddings = False  # as in train_3layer.py
        self.tok = get_tokenizer()
        self.heldout = T.heldout_batches(self.tok)          # 8 x (4, 128)
        # "step0" = the fresh seed-0 init (deterministic, report.md), kept on CPU
        self.states = {"step0": {k: _to_cpu(v) for k, v in
                                 T.trainable_state(self.encoder, self.embedding2).items()}}
        self.ckpt_info = {"step0": {"step": 0, "source": "fresh init via load_all, seed 0"}}
        self.missing = []
        if load_ckpts:
            for name, fn in CKPT_FILES.items():
                path = os.path.join(CKPT_DIR, fn)
                if not os.path.exists(path):
                    print(f"MISSING checkpoint {name}: {path} -- continuing without it")
                    self.missing.append(name)
                    continue
                ck = torch.load(path, map_location="cpu")
                self.states[name] = ck["trainable"]
                self.ckpt_info[name] = {"step": ck["step"], "file": path,
                                        "best_recorded_in_ckpt": ck["best"]}
                del ck
        self.names = list(self.states)
        self.current = "step0"

    def use(self, name):
        state = {k: _to_device(v, self.device) for k, v in self.states[name].items()}
        T.load_trainable_state(self.encoder, self.embedding2, state)
        self.current = name

    @torch.no_grad()
    def teacher_logits(self, ids):
        return self.teacher(input_ids=ids).logits

    @torch.no_grad()
    def gists(self, batches):
        """[N, 16, hidden] gists from the CURRENT checkpoint, generated batch by batch as in training."""
        return torch.cat([generate_softprompt(self.encoder, self.embedding2, b.to(self.device), L)
                          for b in batches])

    @torch.no_grad()
    def per_prompt_kd(self, prefix, ids, teacher_logits=None):
        """Per-prompt training KD loss for a (B,16,H) prefix before (B,128) prompts."""
        if teacher_logits is None:
            teacher_logits = self.teacher_logits(ids)
        student = receiver_forward(self.receiver, prefix, ids)
        return [kd_loss(student[i:i + 1], teacher_logits[i:i + 1]).item() for i in range(ids.shape[0])]

    def extra_batches(self, n_prompts=N_EXTRA, batch_size=4):
        """Up to n_prompts more prompts from the held-out shard, AFTER the 32 train_3layer.py uses.

        Same packing and URL swap as train_3layer.heldout_batches (which only returns the first 8
        batches): the stream is advanced past those 8 batches, so the extra prompts are disjoint from
        the 32 held-out ones, and shard 001_00000 is never read by training at all.
        """
        original = smoke_train.FINEWEB_EDU_SHARD_URL
        smoke_train.FINEWEB_EDU_SHARD_URL = T.HELDOUT_SHARD_URL
        try:
            gen = smoke_train.fineweb_edu_batches(self.tok, T.PROMPT_LEN, batch_size=batch_size)
            first = [next(gen) for _ in range(T.EVAL_BATCHES)]
            extra = [next(gen) for _ in range(n_prompts // batch_size)]
            gen.close()
        finally:
            smoke_train.FINEWEB_EDU_SHARD_URL = original
        assert all(torch.equal(a, b) for a, b in zip(first, self.heldout)), "held-out stream mismatch"
        return extra

    # ---- reference prefixes (all (16, H), embedded with the RECEIVER's embed_tokens) ----
    def emb(self, ids):
        return self.receiver.model.embed_tokens.weight[torch.as_tensor(ids, device=self.device)].detach()

    def text16_ids(self):
        ids = self.tok(TEXT16, add_special_tokens=False)["input_ids"]
        if len(ids) < L:  # extend by repeating the sentence's tokens (not needed for this sentence)
            ids = (ids * (L // len(ids) + 1))[:L]
        return ids[:L]

    def reference_prefixes(self):
        hidden = self.receiver.config.hidden_size
        refs = {"text16": self.emb(self.text16_ids()),
                "zeros": torch.zeros(L, hidden, device=self.device),
                "pad16": self.emb([self.tok.pad_token_id] * L)}
        n_regular = min(self.tok.all_special_ids)  # ids below the first special token are ordinary tokens
        self.rand_tok_ids = {}
        for s in (0, 1, 2):
            g = torch.Generator().manual_seed(s)
            ids = torch.randint(0, n_regular, (L,), generator=g).tolist()
            self.rand_tok_ids[s] = ids
            refs[f"rand_tok_s{s}"] = self.emb(ids)
        return refs

    def reference_info(self):
        return {
            "text16_sentence": TEXT16,
            "text16_ids": self.text16_ids(),
            "text16_tokens": [self.tok.decode([i]) for i in self.text16_ids()],
            "pad16_token": {"token": self.tok.pad_token, "id": self.tok.pad_token_id},
            "rand_tok": {f"seed{s}": {"ids": ids, "tokens": [self.tok.decode([i]) for i in ids]}
                         for s, ids in getattr(self, "rand_tok_ids", {}).items()},
            "rand_tok_range": f"[0, {min(self.tok.all_special_ids)}) (ordinary tokens, specials excluded)",
        }

    def info(self):
        return {"checkpoints": self.ckpt_info, "missing_checkpoints": self.missing,
                "heldout_shard": T.HELDOUT_SHARD_URL, "train_shard": smoke_train.FINEWEB_EDU_SHARD_URL,
                "L": L, "prompt_len": T.PROMPT_LEN, "dtype": "float32", "model": MODEL_PATH,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "base_seed": SEED}


def _to_cpu(v):
    if isinstance(v, dict):
        return {k: _to_cpu(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_cpu(x) for x in v]
    return v.detach().cpu().clone() if torch.is_tensor(v) else v


def _to_device(v, device):
    if isinstance(v, dict):
        return {k: _to_device(x, device) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_device(x, device) for x in v]
    return v.to(device) if torch.is_tensor(v) else v


def batch_split(t, size=4):
    return [t[i:i + size] for i in range(0, t.shape[0], size)]


def derangement(n, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        p = torch.randperm(n, generator=g)
        if not (p == torch.arange(n)).any():
            return p.tolist()
