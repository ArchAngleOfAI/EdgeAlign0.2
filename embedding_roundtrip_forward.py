"""Forward-pass test: Embedding2 -> Transformer(28 layers) -> LM_head -> Embedding1.

Architecture (per user's spec):
- Embedding2: an independent nn.Embedding, initialized by cloning the
  pretrained embed_tokens weight, but NOT tied to anything else (its own
  separate parameter -- a plain copy, not a reference).
- Transformer: the pretrained Qwen3-0.6B decoder stack (28 layers + final
  norm), weights loaded from the checkpoint, unchanged.
- LM_head: unchanged -- remains tied to the ORIGINAL pretrained embedding
  weight (call it Embedding1 / W1). Swapping in Embedding2 as the input
  embedding does not affect this, since lm_head.weight already holds its
  own reference to W1 from the checkpoint's tie_weights() call.
- Embedding1: reusing that same tied weight matrix W1 (vocab_size,
  hidden_size) again, this time as a linear projection from vocab space
  back to hidden space, applied AFTER a softmax over the LM_head's output
  logits: final = softmax(logits) @ W1 (confirmed 2026-09-18 -- raw,
  unnormalized logits @ W1 was tried first and produces huge, unbounded
  magnitudes; softmax first makes this a genuine convex "soft lookup"
  over W1's rows, at a sane, bounded scale).

Forward-pass only (no loss/backward), on one real FineWeb-Edu batch.
"""

import torch
from transformers import AutoTokenizer

from smoke_train import MODEL_PATH, SEQ_LEN, BATCH_SIZE, fineweb_edu_batches
from qwen_dual_embedding import build_frozen_qwen_with_trainable_embedding2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, embedding2, w1, frozen_count, trainable_count = \
        build_frozen_qwen_with_trainable_embedding2(device)
    model.eval()

    print("embedding2 is independent (different tensor from W1):",
          embedding2.weight.data_ptr() != w1.data_ptr())
    print("lm_head still tied to original W1 after swap:",
          model.lm_head.weight.data_ptr() == w1.data_ptr())
    print(f"frozen params: {frozen_count:,}  trainable params (Embedding2): {trainable_count:,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    batch_iter = fineweb_edu_batches(tokenizer, SEQ_LEN, BATCH_SIZE)
    input_ids = next(batch_iter).to(device)
    print("input_ids shape:", tuple(input_ids.shape))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (batch, seq, vocab), via Embedding2 -> transformer -> LM_head
        print("logits shape:", tuple(logits.shape),
              "has_nan:", torch.isnan(logits).any().item())

        probs = torch.softmax(logits, dim=-1)
        final = probs @ w1  # (batch, seq, vocab) @ (vocab, hidden) -> (batch, seq, hidden)
        print("final output shape:", tuple(final.shape),
              "has_nan:", torch.isnan(final).any().item())
        print(f"final output stats: mean={final.mean().item():.4f} std={final.std().item():.4f}")

    print("forward pass succeeded.")


if __name__ == "__main__":
    main()
