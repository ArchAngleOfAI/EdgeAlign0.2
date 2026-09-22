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
   Transformer(frozen, except its last decoder layer) -> LM_head(untied,
   trainable) -> Embedding2(trainable, OUTPUT projection). Used by the
   distillation encoder (distill_softprompt.py). Real tokens are embedded
   by the original frozen Embedding1.

   Per user request (2026-09-22): in addition to Embedding2, the LM_head
   and the transformer's last decoder layer are now also trainable.
   LM_head is TIED to Embedding1 in this checkpoint
   (tie_word_embeddings=True, confirmed: model.lm_head.weight IS
   model.model.embed_tokens.weight, same data_ptr) -- flipping
   requires_grad on it directly would also make Embedding1 (the frozen
   real-token input embedding, a deliberate 2026-09-18 correction) train.
   Per the user's explicit choice, LM_head is instead UNTIED first (given
   its own cloned, independent Parameter) so it can train while Embedding1
   stays frozen exactly as before.
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
    """Embedding1(frozen, input) -> Transformer(frozen except last layer) ->
    LM_head(untied, trainable) -> Embedding2(trainable, output).

    Returns (model, embedding2, w1): `model` is a Qwen3ForCausalLM with
    embed_tokens (Embedding1, real-token input embedding) frozen and never
    swapped, but its LAST decoder layer and its LM_head both trainable
    (LM_head is untied from Embedding1 first -- see module docstring for
    why). `embedding2` is a separate, independent nn.Embedding, initialized
    as a clone of `w1` (= the ORIGINAL tied lm_head/embed_tokens weight,
    captured before untying) -- also trainable, used purely as an
    output-side projection (never as the input embedding).
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

    # Untie LM_head from Embedding1 (embed_tokens) so it can train without
    # also making the frozen real-token input embedding trainable.
    model.lm_head.weight = nn.Parameter(w1.detach().clone())
    model.lm_head.weight.requires_grad = True

    for param in model.model.layers[-1].parameters():
        param.requires_grad = True

    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = (embedding2.weight.numel()
                        + sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model, embedding2, w1, frozen_count, trainable_count
