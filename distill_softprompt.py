"""Soft-prompt distillation setup, per user's spec (confirmed 2026-09-18):

- Teacher stream: a plain frozen Qwen3-0.6B instance. Sees the real
  prompt tokens directly, produces the target output distribution.
- Student encoder ("softprompt generator", corrected 2026-09-18): built
  by qwen_dual_embedding.build_softprompt_generator -- Embedding1(frozen,
  input) -> Transformer(frozen) -> LM_head(frozen) -> Embedding2
  (trainable, output). The REAL prompt tokens are embedded by the
  original frozen Embedding1 (not Embedding2 -- that was the earlier,
  now-corrected version). Only the repeated logits->hidden projection
  step (which also produces each step's fed-back input) is trainable.
  Autoregressive CONTINUOUS rollout, no discretization at any step: the
  Embedding2-projected output for the last position becomes the next
  step's input embedding directly, via inputs_embeds + KV cache. This
  sequence of 100 hidden-size vectors is the "softprompt".
- Student receiver: a SEPARATE frozen Qwen3-0.6B instance (its own load,
  distinct from the teacher). Conceptually its input is
  [100 x MASK/BLANK token] + [real prompt tokens]; the MASK positions'
  looked-up embeddings get replaced by the softprompt. Implemented here
  as torch.cat([softprompt, receiver.model.embed_tokens(prompt_ids)]) --
  mathematically identical to "embed then overwrite", but avoids an
  in-place tensor mutation that would break the autograd path back into
  Embedding2 (in-place slice-assignment into a non-grad tensor does not
  reliably propagate gradients; concatenation does).
- Loss: KL(teacher_probs || student_probs) over the vocab, computed at
  the receiver's output positions aligned to the real prompt (i.e.
  positions [100:], skipping the soft-prefix positions), against the
  teacher's output at its own prompt positions [0:]. Temperature = 1.0
  (not specified by the user -- flagged default, easy to add as a flag
  later if needed).
- Only Embedding2 receives gradients / gets optimized (AdamW, lr=5e-5
  flat, matching the convention used in the earlier smoke tests).
  Everything else (teacher, receiver, and the encoder's own frozen
  transformer+LM_head) stays frozen throughout.
- Position IDs: no manual override. With no past_key_values, HF assigns
  position_ids = 0..seq_len-1 to whatever `inputs_embeds` sequence it's
  given -- so for the receiver's [softprompt ++ prompt] sequence, the
  real prompt tokens naturally land on positions 100..100+N-1 (i.e.
  "continue from the soft prefix", per the user's choice), with no
  special-casing needed.

PROMPT_LEN (128) is a default not specified by the user -- distinct from
the SEQ_LEN=1024 used in the earlier plain-LM smoke tests, chosen here to
keep the 100-step autoregressive softprompt rollout + receiver forward
manageable for a first correctness check. NUM_STEPS=10, matching the
"smoke test" scale used throughout this session so far.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from smoke_train import MODEL_PATH, fineweb_edu_batches
from qwen_dual_embedding import build_softprompt_generator

PROMPT_LEN = 128
NUM_SOFT_TOKENS = 100
LR = 5e-5
NUM_STEPS = 10
KD_TEMPERATURE = 1.0


def generate_softprompt(encoder_model, embedding2, prompt_ids, num_soft_tokens):
    """Autoregressive CONTINUOUS rollout: no token is ever sampled/discretized.

    encoder_model is fully frozen (Embedding1/Transformer/LM_head); only
    embedding2 (passed separately, never part of encoder_model) is
    trainable, used purely as the output-side logits->hidden projection.

    Returns (batch, num_soft_tokens, hidden_size).
    """
    inner = encoder_model.model  # Qwen3Model, frozen -- embed_tokens here IS Embedding1

    embeds = inner.embed_tokens(prompt_ids)  # frozen Embedding1
    out = inner(inputs_embeds=embeds, use_cache=True)
    past_key_values = out.past_key_values
    hidden = out.last_hidden_state[:, -1:, :]  # (batch, 1, hidden)

    soft_tokens = []
    for step in range(num_soft_tokens):
        logits = encoder_model.lm_head(hidden)          # (batch, 1, vocab) -- frozen LM_head
        probs = torch.softmax(logits, dim=-1)           # convex weights over vocab rows
        soft_token = probs @ embedding2.weight          # (batch, 1, hidden) -- TRAINABLE Embedding2
        soft_tokens.append(soft_token)
        if step == num_soft_tokens - 1:
            break
        out = inner(inputs_embeds=soft_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        hidden = out.last_hidden_state

    return torch.cat(soft_tokens, dim=1)


def receiver_forward(receiver_model, softprompt, prompt_ids):
    """[softprompt ++ prompt] through the frozen receiver; returns logits at prompt positions."""
    prompt_embeds = receiver_model.model.embed_tokens(prompt_ids)  # frozen Embedding1
    combined = torch.cat([softprompt, prompt_embeds], dim=1)
    out = receiver_model.model(inputs_embeds=combined)
    logits_full = receiver_model.lm_head(out.last_hidden_state)
    return logits_full[:, softprompt.shape[1]:, :]  # aligned to real prompt positions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_encoder, embedding2, w1, frozen_count, trainable_count = \
        build_softprompt_generator(device)

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    receiver = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    for p in receiver.parameters():
        p.requires_grad = False
    teacher.eval()
    receiver.eval()

    optimizer = torch.optim.AdamW([embedding2.weight], lr=LR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    batch_iter = fineweb_edu_batches(tokenizer, PROMPT_LEN, batch_size=1)

    print(f"device={device}, prompt_len={PROMPT_LEN}, num_soft_tokens={NUM_SOFT_TOKENS}, "
          f"lr={LR}, kd_temperature={KD_TEMPERATURE}")
    print(f"frozen params (x2 instances + encoder backbone): ~{frozen_count:,} each  "
          f"trainable (Embedding2): {trainable_count:,}")

    for step in range(1, NUM_STEPS + 1):
        prompt_ids = next(batch_iter).to(device)

        softprompt = generate_softprompt(student_encoder, embedding2, prompt_ids, NUM_SOFT_TOKENS)

        with torch.no_grad():
            teacher_logits = teacher(input_ids=prompt_ids).logits

        student_logits = receiver_forward(receiver, softprompt, prompt_ids)

        student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=-1)
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        loss = kl_per_token.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step:2d}/{NUM_STEPS}  kd_loss={loss.item():.6f}")

    print("done.")


if __name__ == "__main__":
    main()
