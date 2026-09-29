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

import os

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
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_EVERY = 10_000


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
    import argparse
    import sys

    # Redirected stdout (e.g. `> log.txt`) is fully block-buffered by
    # default, not line-buffered -- with short per-step lines that can mean
    # NO output reaches the log file until thousands of lines have
    # accumulated, or the process exits. Force line buffering so progress
    # (and a crash traceback) shows up in the log as it happens.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=NUM_STEPS,
                         help="Number of OPTIMIZER steps (each consists of --grad-accum-steps "
                              "micro-batches accumulated before a single optimizer.step()).")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Per-micro-batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                         help="Micro-batches accumulated per optimizer step. Global batch size "
                              "= batch-size * grad-accum-steps. Only student_encoder's trainable "
                              "params (embedding2, lm_head, last layer) ever accumulate gradients "
                              "-- teacher and receiver are fully frozen, so this is inherently "
                              "grad accumulation on the encoder side only.")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint .pt file to resume from.")
    args = parser.parse_args()
    num_steps = args.steps
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps
    checkpoint_every = args.checkpoint_every
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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

    # Trainable set (2026-09-22): embedding2 + the student encoder's untied
    # LM_head + its transformer's last decoder layer -- see
    # qwen_dual_embedding.build_softprompt_generator for what's frozen vs not.
    trainable_params = (
        [embedding2.weight, student_encoder.lm_head.weight]
        + list(student_encoder.model.layers[-1].parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    start_step = 0
    previous_checkpoint_path = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        embedding2.load_state_dict(ckpt["embedding2_state_dict"])
        student_encoder.lm_head.load_state_dict(ckpt["lm_head_state_dict"])
        student_encoder.model.layers[-1].load_state_dict(ckpt["last_layer_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        previous_checkpoint_path = args.resume
        print(f"resumed from {args.resume} at step {start_step}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    batch_iter = fineweb_edu_batches(tokenizer, PROMPT_LEN, batch_size=batch_size)

    global_batch_size = batch_size * grad_accum_steps
    print(f"device={device}, prompt_len={PROMPT_LEN}, batch_size={batch_size}, "
          f"grad_accum_steps={grad_accum_steps}, global_batch_size={global_batch_size}, "
          f"num_soft_tokens={NUM_SOFT_TOKENS}, lr={LR}, kd_temperature={KD_TEMPERATURE}, "
          f"checkpoint_every={checkpoint_every}")
    print(f"frozen params (x2 instances + encoder backbone): ~{frozen_count:,} each  "
          f"trainable (embedding2 + lm_head + last layer): {trainable_count:,}")

    for step in range(start_step + 1, num_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                prompt_ids = next(batch_iter).to(device)

                softprompt = generate_softprompt(student_encoder, embedding2, prompt_ids, NUM_SOFT_TOKENS)

                with torch.no_grad():
                    teacher_logits = teacher(input_ids=prompt_ids).logits

                student_logits = receiver_forward(receiver, softprompt, prompt_ids)

                student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1)
                teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=-1)
                kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
                micro_loss = kl_per_token.mean()

                # Scaled so the accumulated .grad matches the mean loss over the
                # whole global batch, not the sum over micro-batches.
                (micro_loss / grad_accum_steps).backward()
                accum_loss += micro_loss.item()
            except torch.OutOfMemoryError as e:
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 1e9
                    reserved = torch.cuda.memory_reserved(device) / 1e9
                    print(f"OOM at step {step}/{num_steps} micro-batch {micro_step + 1}/"
                          f"{grad_accum_steps}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB "
                          f"error={e}", flush=True)
                raise

        optimizer.step()
        loss_value = accum_loss / grad_accum_steps

        print(f"step {step:3d}/{num_steps}  kd_loss={loss_value:.6f}"
              + (f"  (avg over {grad_accum_steps} micro-batches)" if grad_accum_steps > 1 else ""))

        if step % checkpoint_every == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"embedding2_step{step}.pt")
            torch.save({
                "step": step,
                "embedding2_state_dict": embedding2.state_dict(),
                "lm_head_state_dict": student_encoder.lm_head.state_dict(),
                "last_layer_state_dict": student_encoder.model.layers[-1].state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            if previous_checkpoint_path is not None and os.path.exists(previous_checkpoint_path):
                os.remove(previous_checkpoint_path)
            previous_checkpoint_path = ckpt_path
            print(f"saved checkpoint: {ckpt_path} (previous checkpoint deleted)")

    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1e9
        reserved = torch.cuda.max_memory_reserved(device) / 1e9
        print(f"peak GPU memory: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")

    print("done.")


if __name__ == "__main__":
    main()
