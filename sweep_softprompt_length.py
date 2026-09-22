"""Diagnostic sweep: initial (no-training) KD loss vs. soft-prompt length.

Sanity check requested by the user: since KD alignment was confirmed correct
(see distill_softprompt.py's docstring/derivation), the intuition is that a
LONGER soft-prompt should carry more of the real prompt's information and
thus produce a LOWER initial KD loss (closer to the teacher, which sees the
real prompt directly) -- this script measures that curve.

No training happens here. All three model instances (encoder/teacher/
receiver) use the pretrained Qwen3-0.6B checkpoint as-is, exactly like
build_softprompt_generator's default init (embedding2 starts as a clone of
the tied lm_head/embedding weight W1) -- so "initial" here means "before any
optimizer step ever touches embedding2", same starting point as step 1 of
distill_softprompt.py.

Efficiency note: the softprompt rollout is causal/autoregressive, so soft
token i only depends on soft tokens 0..i-1 and the prompt -- generating a
length-200 rollout ONCE and taking prefixes [:L] for L=1..200 gives
identical values to 200 separate from-scratch rollouts. We exploit that here
(one rollout per prompt, not 200), but the receiver forward pass at each L
is still recomputed from scratch (simplest correct approach; flagged here in
case 200 recomputes/prompt turns out too slow and KV-cache reuse across L is
worth adding later).

NUM_PROMPTS=5 (averaged, with std shown as a shaded band) is a default not
specified by the user -- flagged since a single prompt's curve could be
noisy; adjust via --num-prompts.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from smoke_train import MODEL_PATH, fineweb_edu_batches
from qwen_dual_embedding import build_softprompt_generator
from distill_softprompt import PROMPT_LEN, KD_TEMPERATURE

MAX_SOFT_TOKENS = 200
NUM_PROMPTS = 5
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "kd_loss_vs_softprompt_length.png")


def generate_softprompt_full(encoder_model, embedding2, prompt_ids, max_len):
    """Same rollout as distill_softprompt.generate_softprompt, but returns
    ALL intermediate soft tokens up to max_len so callers can take prefixes.
    """
    inner = encoder_model.model

    embeds = inner.embed_tokens(prompt_ids)
    out = inner(inputs_embeds=embeds, use_cache=True)
    past_key_values = out.past_key_values
    hidden = out.last_hidden_state[:, -1:, :]

    soft_tokens = []
    for step in range(max_len):
        logits = encoder_model.lm_head(hidden)
        probs = torch.softmax(logits, dim=-1)
        soft_token = probs @ embedding2.weight
        soft_tokens.append(soft_token)
        if step == max_len - 1:
            break
        out = inner(inputs_embeds=soft_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        hidden = out.last_hidden_state

    return torch.cat(soft_tokens, dim=1)  # (batch, max_len, hidden)


def receiver_forward(receiver_model, softprompt, prompt_ids):
    """[softprompt ++ prompt] through the frozen receiver; returns logits at prompt positions."""
    prompt_embeds = receiver_model.model.embed_tokens(prompt_ids)
    combined = torch.cat([softprompt, prompt_embeds], dim=1)
    out = receiver_model.model(inputs_embeds=combined)
    logits_full = receiver_model.lm_head(out.last_hidden_state)
    return logits_full[:, softprompt.shape[1]:, :]


def kd_loss(student_logits, teacher_logits):
    student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return kl_per_token.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=MAX_SOFT_TOKENS,
                         help="Max soft-prompt length to sweep up to (sweeps 1..max-len).")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS,
                         help="Number of FineWeb-Edu prompts to average over.")
    parser.add_argument("--output", type=str, default=OUTPUT_PNG)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_encoder, embedding2, w1, _, _ = build_softprompt_generator(device)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    receiver = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    for m in (teacher, receiver):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    batch_iter = fineweb_edu_batches(tokenizer, PROMPT_LEN, batch_size=1)

    lengths = list(range(1, args.max_len + 1))
    print(f"device={device}, prompt_len={PROMPT_LEN}, max_soft_len={args.max_len}, "
          f"num_prompts={args.num_prompts}, kd_temperature={KD_TEMPERATURE}")

    losses_per_prompt = []
    with torch.no_grad():
        for prompt_idx in range(args.num_prompts):
            prompt_ids = next(batch_iter).to(device)
            teacher_logits = teacher(input_ids=prompt_ids).logits

            full_softprompt = generate_softprompt_full(
                student_encoder, embedding2, prompt_ids, args.max_len)

            per_length = []
            for L in lengths:
                softprompt_L = full_softprompt[:, :L, :]
                student_logits = receiver_forward(receiver, softprompt_L, prompt_ids)
                per_length.append(kd_loss(student_logits, teacher_logits))
            losses_per_prompt.append(per_length)
            print(f"prompt {prompt_idx + 1}/{args.num_prompts} done "
                  f"(L=1 loss={per_length[0]:.4f}, L={args.max_len} loss={per_length[-1]:.4f})")

    losses = torch.tensor(losses_per_prompt)  # (num_prompts, len(lengths))
    mean_losses = losses.mean(dim=0)
    std_losses = losses.std(dim=0) if args.num_prompts > 1 else torch.zeros_like(mean_losses)

    for L, m, s in zip(lengths, mean_losses.tolist(), std_losses.tolist()):
        print(f"soft_prompt_len={L:3d}  kd_loss_mean={m:.6f}  kd_loss_std={s:.6f}")

    plt.figure(figsize=(9, 5.5))
    plt.plot(lengths, mean_losses.tolist(), marker="o", markersize=2, linewidth=1)
    if args.num_prompts > 1:
        plt.fill_between(lengths,
                          (mean_losses - std_losses).tolist(),
                          (mean_losses + std_losses).tolist(),
                          alpha=0.2)
    plt.xlabel("Soft-prompt length (tokens)")
    plt.ylabel(f"Initial KD loss (KL div, T={KD_TEMPERATURE})")
    plt.title("Initial KD loss vs. soft-prompt length\n"
               f"(pretrained weights, no training, averaged over {args.num_prompts} "
               "FineWeb-Edu prompts)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()
