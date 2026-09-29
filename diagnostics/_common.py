"""Shared setup for the flat-KD-loss diagnostics.

Everything here REUSES the training code rather than reimplementing it:
generate_softprompt / receiver_forward / constants come from
distill_softprompt.py, the model builder from qwen_dual_embedding.py, and
the data stream from smoke_train.py. The only thing that had to be copied is
the KD-loss expression, because distill_softprompt.py computes it inline in
main() (not as an importable function) -- `kd_loss` below is those exact
lines, verbatim.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from distill_softprompt import (  # noqa: E402
    KD_TEMPERATURE, LR, NUM_SOFT_TOKENS, PROMPT_LEN, generate_softprompt, receiver_forward,
)
from qwen_dual_embedding import build_softprompt_generator  # noqa: E402
from smoke_train import MODEL_PATH, fineweb_edu_batches  # noqa: E402

# The flat 17k-step run (logs/distill_100k.log) used --batch-size 4, no grad accumulation.
BATCH_SIZE = 4
SEED = 0


def setup_stdout():
    sys.stdout.reconfigure(line_buffering=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all(device, num_trainable_layers=1):
    """Student encoder + embedding2 (via build_softprompt_generator), frozen teacher and receiver.

    num_trainable_layers > 1 additionally unfreezes the encoder's last N decoder
    layers -- done here, after the builder returns, so qwen_dual_embedding.py
    itself is untouched.
    """
    torch.manual_seed(SEED)
    encoder, embedding2, w1, _, _ = build_softprompt_generator(device)
    for layer in encoder.model.layers[-num_trainable_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    receiver = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    for p in receiver.parameters():
        p.requires_grad = False
    teacher.eval()
    receiver.eval()
    return encoder, embedding2, w1, teacher, receiver


def trainable_params(encoder, embedding2, num_trainable_layers=1):
    """Same list distill_softprompt.main() builds, generalized to the last N layers."""
    params = [embedding2.weight, encoder.lm_head.weight]
    for layer in encoder.model.layers[-num_trainable_layers:]:
        params += list(layer.parameters())
    return params


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


def batch_stream(tokenizer, batch_size=BATCH_SIZE):
    """The exact stream distill_softprompt.py consumes (deterministic order)."""
    return fineweb_edu_batches(tokenizer, PROMPT_LEN, batch_size=batch_size)


def take_batches(tokenizer, n, skip=0, batch_size=BATCH_SIZE):
    it = batch_stream(tokenizer, batch_size)
    for _ in range(skip):
        next(it)
    return [next(it) for _ in range(n)]


def kd_loss(student_logits, teacher_logits):
    """Verbatim from distill_softprompt.main()."""
    student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return kl_per_token.mean()


def forward_loss(encoder, embedding2, teacher, receiver, prompt_ids, num_soft_tokens=NUM_SOFT_TOKENS):
    """One micro-batch forward exactly as in distill_softprompt.main()."""
    softprompt = generate_softprompt(encoder, embedding2, prompt_ids, num_soft_tokens)
    with torch.no_grad():
        teacher_logits = teacher(input_ids=prompt_ids).logits
    student_logits = receiver_forward(receiver, softprompt, prompt_ids)
    return kd_loss(student_logits, teacher_logits)


@torch.no_grad()
def eval_loss(encoder, embedding2, teacher, receiver, batches, device):
    """Mean KD loss over a fixed list of batches (no grad, no param change)."""
    vals = [forward_loss(encoder, embedding2, teacher, receiver, b.to(device)).item() for b in batches]
    return sum(vals) / len(vals), vals


def snapshot(params):
    return [p.detach().clone() for p in params]


def rel_change(params, snap):
    num = sum(((p.detach() - s) ** 2).sum() for p, s in zip(params, snap)).sqrt()
    den = sum((s ** 2).sum() for s in snap).sqrt()
    return (num / den).item(), num.item()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
