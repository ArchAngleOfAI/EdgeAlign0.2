"""Test 6 (Hypothesis B): does unfreezing more encoder layers make the loss move?

Trains with the last N decoder layers trainable (default 3; the unfreezing is
done in _common.load_all, not in qwen_dual_embedding.py) for up to --max-steps
optimizer steps or --time-limit seconds of training, whichever comes first.
Loop, data stream, LR (5e-5), optimizer and batch size (4) match the flat
17k-step run; the KD loss is printed every step.

Because per-step training loss is dominated by batch-to-batch variance (std
~0.2 in logs/distill_100k.log), the script also measures a paired, held-out
eval loss on the same 8 fixed batches (stream batches 1001-1008, never seen in
training here) before and after training. Run with --num-layers 1 for the
matched control.
"""

import argparse
import statistics

import torch

from _common import (
    LR, Timer, eval_loss, forward_loss, get_device, get_tokenizer, load_all, rel_change,
    setup_stdout, snapshot, take_batches, trainable_params, batch_stream,
)

EVAL_SKIP, EVAL_N = 1000, 8


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--time-limit", type=float, default=15 * 60)
    args = ap.parse_args()

    device = get_device()
    tok = get_tokenizer()
    encoder, embedding2, _, teacher, receiver = load_all(device, num_trainable_layers=args.num_layers)
    params = trainable_params(encoder, embedding2, args.num_layers)
    print(f"trainable layers: last {args.num_layers}; trainable params: {sum(p.numel() for p in params):,}")
    eval_batches = take_batches(tok, EVAL_N, skip=EVAL_SKIP)

    t = Timer()
    eval_before, per_before = eval_loss(encoder, embedding2, teacher, receiver, eval_batches, device)
    print(f"eval BEFORE: {eval_before:.6f}  ({t():.0f}s)")

    snap = snapshot(params)
    opt = torch.optim.AdamW(params, lr=LR)
    stream = batch_stream(tok)
    losses, t, limit = [], Timer(), "max-steps"
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        loss = forward_loss(encoder, embedding2, teacher, receiver, next(stream).to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"step {step:3d}  kd_loss={losses[-1]:.6f}  ({t():.0f}s)")
        if t() >= args.time_limit:
            limit = "time-limit"
            break
    train_time = t()
    rc, _ = rel_change(params, snap)

    t = Timer()
    eval_after, per_after = eval_loss(encoder, embedding2, teacher, receiver, eval_batches, device)
    print(f"eval AFTER: {eval_after:.6f}  ({t():.0f}s)")

    n = len(losses)
    q = max(n // 4, 1)
    diffs = [a - b for a, b in zip(per_after, per_before)]
    print(f"\nSUMMARY num_layers={args.num_layers}: {n} steps in {train_time:.0f}s "
          f"(limit hit: {limit}), ||dtheta||/||theta||={rc:.3e}")
    print(f"  train loss mean first {q} steps={statistics.mean(losses[:q]):.4f}, "
          f"last {q} steps={statistics.mean(losses[-q:]):.4f}")
    print(f"  held-out eval ({EVAL_N} fixed batches): before={eval_before:.4f} after={eval_after:.4f} "
          f"paired diff={statistics.mean(diffs):+.4f} (improved on {sum(d < 0 for d in diffs)}/{EVAL_N})")


if __name__ == "__main__":
    main()
