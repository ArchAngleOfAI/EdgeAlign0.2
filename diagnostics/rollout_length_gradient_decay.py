"""Test 7 (Hypothesis B): does gradient reaching embedding2 vanish as the rollout gets longer?

Part 1 (as specified): same single forward+backward as check_gradients.py, on
the same first batch, with generate_softprompt truncated to L = 10, 30, 60, 100
soft tokens (the receiver therefore sees an L-token prefix). Reports ||grad||
of embedding2 (plus lm_head / last layer / loss for context).
Caveat: truncation changes BOTH the depth of backprop-through-time AND the
number of soft tokens feeding the receiver directly, so part 1 alone can't
separate "deeper BPTT" from "more direct paths".

Part 2 (added to separate the two): a COPY of generate_softprompt at L=100
with a switch to detach the fed-back soft token before it enters the next
rollout step. Detached = only the DIRECT path (soft token -> receiver) carries
gradient; full = direct + recurrent (BPTT) paths. Per-position ||dL/ds_k||
from both runs shows how much gradient arrives through the recurrence at each
depth, and how much of embedding2's gradient comes from BPTT at all.
"""

import torch

from _common import (
    generate_softprompt, get_device, get_tokenizer, kd_loss, load_all, receiver_forward,
    setup_stdout, take_batches,
)

LENGTHS = (10, 30, 60, 100)


def grad_norms(encoder, embedding2):
    last = sum((p.grad ** 2).sum() for p in encoder.model.layers[-1].parameters()).sqrt().item()
    return embedding2.weight.grad.norm().item(), encoder.lm_head.weight.grad.norm().item(), last


def zero(encoder, embedding2):
    embedding2.weight.grad = None
    for p in encoder.parameters():
        p.grad = None


def generate_softprompt_probe(encoder_model, embedding2, prompt_ids, num_soft_tokens, detach_feedback):
    """generate_softprompt with retain_grad on each soft token and optional feedback detach."""
    inner = encoder_model.model
    out = inner(inputs_embeds=inner.embed_tokens(prompt_ids), use_cache=True)
    past_key_values = out.past_key_values
    hidden = out.last_hidden_state[:, -1:, :]
    soft_tokens = []
    for step in range(num_soft_tokens):
        probs = torch.softmax(encoder_model.lm_head(hidden), dim=-1)
        soft_token = probs @ embedding2.weight
        soft_token.retain_grad()
        soft_tokens.append(soft_token)
        if step == num_soft_tokens - 1:
            break
        fed = soft_token.detach() if detach_feedback else soft_token
        out = inner(inputs_embeds=fed, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        hidden = out.last_hidden_state
    return torch.cat(soft_tokens, dim=1), soft_tokens


def main():
    setup_stdout()
    device = get_device()
    encoder, embedding2, _, teacher, receiver = load_all(device)
    ids = take_batches(get_tokenizer(), 1)[0].to(device)
    with torch.no_grad():
        teacher_logits = teacher(input_ids=ids).logits

    print("PART 1: truncated rollout length vs gradient norm")
    print(f"{'L':>4s} {'kd_loss':>10s} {'|g emb2|':>12s} {'|g lm_head|':>12s} {'|g last layer|':>15s} "
          f"{'|g emb2|/L':>12s}")
    for L in LENGTHS:
        zero(encoder, embedding2)
        sp = generate_softprompt(encoder, embedding2, ids, L)
        loss = kd_loss(receiver_forward(receiver, sp, ids), teacher_logits)
        loss.backward()
        e, h, l = grad_norms(encoder, embedding2)
        print(f"{L:4d} {loss.item():10.6f} {e:12.4e} {h:12.4e} {l:15.4e} {e / L:12.4e}")

    print("\nPART 2: L=100, full BPTT vs feedback detached (direct paths only)")
    per_pos = {}
    for mode in ("full", "detached"):
        zero(encoder, embedding2)
        sp, toks = generate_softprompt_probe(encoder, embedding2, ids, 100, detach_feedback=(mode == "detached"))
        loss = kd_loss(receiver_forward(receiver, sp, ids), teacher_logits)
        loss.backward()
        e, h, l = grad_norms(encoder, embedding2)
        per_pos[mode] = [t.grad.norm().item() for t in toks]
        print(f"  {mode:9s} kd_loss={loss.item():.6f}  |g emb2|={e:.4e}  |g lm_head|={h:.4e}  "
              f"|g last layer|={l:.4e}")
        if mode == "full":
            g_full = embedding2.weight.grad.detach().clone()
        else:
            g_direct = embedding2.weight.grad.detach().clone()
    g_rec = g_full - g_direct
    cos = torch.nn.functional.cosine_similarity(g_full.flatten(), g_direct.flatten(), dim=0).item()
    print(f"  embedding2 grad: |recurrent part| = |full - direct| = {g_rec.norm().item():.4e}  "
          f"(= {g_rec.norm().item() / g_full.norm().item():.1%} of |full|), cos(full, direct)={cos:.4f}")

    print("\n  per-position ||dL/d soft_token_k||  (k=1 is generated first, so has the MOST downstream steps)")
    print(f"  {'k':>4s} {'full':>12s} {'direct only':>12s} {'full/direct':>12s}")
    for k in (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100):
        f, d = per_pos["full"][k - 1], per_pos["detached"][k - 1]
        print(f"  {k:4d} {f:12.4e} {d:12.4e} {f / d:12.3f}")


if __name__ == "__main__":
    main()
