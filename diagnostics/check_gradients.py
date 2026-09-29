"""Test 1 (Hypothesis A): do the trainable params actually receive gradient?

One forward+backward pass exactly as distill_softprompt.py does it (reusing its
generate_softprompt / receiver_forward), on the first training batch. Prints the
L2 norm of every trainable param's .grad and flags None / exactly-zero grads.

Beyond the raw norm it also reports, per param, the fraction of elements whose
|grad| exceeds AdamW's eps (1e-8) -- AdamW normalizes each element by its own
running RMS, so what matters for whether a weight moves is "is the grad nonzero
and above eps", not its absolute size.
"""

import torch

from _common import (
    Timer, get_device, get_tokenizer, load_all, forward_loss, setup_stdout, take_batches,
    trainable_params,
)


def describe(name, p):
    g = p.grad
    if g is None:
        print(f"  {name:55s} grad=None   <-- NO GRADIENT")
        return
    norm = torch.linalg.norm(g).item()
    nz = (g != 0).float().mean().item()
    above_eps = (g.abs() > 1e-8).float().mean().item()
    ratio = norm / max(torch.linalg.norm(p.detach()).item(), 1e-30)
    flag = "   <-- EXACTLY ZERO" if norm == 0.0 else ""
    print(f"  {name:55s} |grad|={norm:.4e}  |grad|/|w|={ratio:.3e}  "
          f"nonzero={nz:.4f}  |g|>1e-8={above_eps:.4f}{flag}")


def main():
    setup_stdout()
    device = get_device()
    t = Timer()
    encoder, embedding2, _, teacher, receiver = load_all(device)
    tok = get_tokenizer()
    prompt_ids = take_batches(tok, 1)[0].to(device)
    print(f"loaded in {t():.1f}s; batch shape {tuple(prompt_ids.shape)}")

    t = Timer()
    loss = forward_loss(encoder, embedding2, teacher, receiver, prompt_ids)
    loss.backward()
    torch.cuda.synchronize()
    print(f"kd_loss={loss.item():.6f}  (logs/distill_100k.log step 1 = 0.446160)  fwd+bwd {t():.1f}s")

    print("\nGradient norms:")
    describe("embedding2.weight", embedding2.weight)
    describe("student_encoder.lm_head.weight", encoder.lm_head.weight)
    last = encoder.model.layers[-1]
    for n, p in last.named_parameters():
        describe(f"student_encoder.model.layers[-1].{n}", p)

    # Frozen params must NOT have gradient (sanity check the other direction).
    last_prefix = f"model.layers.{len(encoder.model.layers) - 1}."
    leaked = [n for n, p in encoder.named_parameters()
              if p.grad is not None and n != "lm_head.weight" and not n.startswith(last_prefix)]
    print(f"\nencoder params OUTSIDE the trainable set that got a .grad: {leaked or 'none'}")
    print(f"embed_tokens (Embedding1) grad: {encoder.model.embed_tokens.weight.grad}")

    # Which embedding2 rows actually get gradient? (probs @ E only reaches rows with prob mass)
    row_norms = embedding2.weight.grad.norm(dim=1)
    print(f"\nembedding2 rows with nonzero grad: {(row_norms > 0).sum().item()} / {row_norms.numel()}")
    top = row_norms.topk(10)
    share = (top.values ** 2).sum() / (row_norms ** 2).sum()
    print(f"top-10 rows hold {share.item():.1%} of embedding2's squared grad norm; "
          f"top tokens: {[tok.decode([i]) for i in top.indices.tolist()]}")

    all_none = all(p.grad is None or p.grad.abs().sum() == 0
                   for p in trainable_params(encoder, embedding2))
    print(f"\nVERDICT: {'BROKEN -- no trainable param has gradient' if all_none else 'all trainable params receive nonzero gradient'}")


if __name__ == "__main__":
    main()
