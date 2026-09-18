"""Builders for the two Qwen3-0.6B dual-embedding architectures used in this project.

Two distinct architectures, distinguished by WHERE the trainable
Embedding2 sits relative to the frozen Embedding1 (the original
pretrained embed_tokens/lm_head weight, W1, tied to LM_head):

1. `build_frozen_qwen_with_trainable_embedding2` -- Embedding2(trainable,
   INPUT) -> Transformer(frozen) -> LM_head(frozen) -> Embedding1(frozen,
   OUTPUT projection via W1). Used by the standalone roundtrip forward-pass
   smoke test (embedding_roundtrip_forward.py). Real tokens are embedded
   by the (still-being-trained) Embedding2.

2. `build_softprompt_generator` -- Embedding1(frozen, INPUT) ->
   Transformer(frozen) -> LM_head(frozen) -> Embedding2(trainable, OUTPUT
   projection). Used by the distillation encoder (distill_softprompt.py).
   Real tokens are embedded by the original frozen Embedding1; only the
   final logits->hidden projection (feeding each generated soft token,
   and the next step's input) is trainable.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_PATH = "/data/models/huggingface/qwen3-0.6b"


def build_frozen_qwen_with_trainable_embedding2(device, dtype=torch.float32):
    """Embedding2(trainable, input) -> Transformer(frozen) -> LM_head(frozen) -> Embedding1(frozen, output)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype)
    model.to(device)

    w1 = model.lm_head.weight  # (vocab_size, hidden_size) -- Embedding1, tied to LM_head
    vocab_size, hidden_size = w1.shape

    embedding2 = nn.Embedding(vocab_size, hidden_size)
    embedding2.weight = nn.Parameter(w1.detach().clone())
    embedding2.to(device)
    model.model.embed_tokens = embedding2

    for name, param in model.named_parameters():
        param.requires_grad = "embed_tokens" in name  # only Embedding2 is embed_tokens now

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == ["model.embed_tokens.weight"], trainable
    assert trainable_count == vocab_size * hidden_size

    return model, embedding2, w1, frozen_count, trainable_count


def build_softprompt_generator(device, dtype=torch.float32):
    """Embedding1(frozen, input) -> Transformer(frozen) -> LM_head(frozen) -> Embedding2(trainable, output).

    Returns (model, embedding2, w1): `model` is a plain, fully-frozen
    Qwen3ForCausalLM (embed_tokens included -- this IS Embedding1, never
    swapped). `embedding2` is a separate, independent nn.Embedding,
    initialized as a clone of `w1` (= model.lm_head.weight) but not tied
    to it -- the only trainable component, used purely as an output-side
    projection (never as the input embedding).
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    w1 = model.lm_head.weight  # (vocab_size, hidden_size) -- Embedding1, tied to LM_head, frozen
    vocab_size, hidden_size = w1.shape

    embedding2 = nn.Embedding(vocab_size, hidden_size)
    embedding2.weight = nn.Parameter(w1.detach().clone())  # independent copy, trainable
    embedding2.to(device)

    frozen_count = sum(p.numel() for p in model.parameters())
    trainable_count = embedding2.weight.numel()

    return model, embedding2, w1, frozen_count, trainable_count
