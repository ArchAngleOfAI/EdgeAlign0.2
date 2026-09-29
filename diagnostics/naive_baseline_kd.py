"""Test 5 (Hypothesis B): how good is the generated softprompt vs. trivial fixed prefixes?

Same KD loss path (receiver_forward + the training loop's KL expression), on the
first 20 batches of the exact FineWeb-Edu stream training uses (batch_size=4,
so these are literally the batches of steps 1-20 of logs/distill_100k.log).
The 100-vector prefix is replaced by:

  (a) 100 copies of the BOS token's Embedding1 vector, or the pad token's if the
      tokenizer has no BOS (which one is printed),
  (b) 100 zero vectors,
and, for reference:
  (c) the UNTRAINED generated softprompt (the training start point),
  (d) the softprompt generated with embedding2 from checkpoints/embedding2_step10000.pt
      -- the only surviving checkpoint of the flat run (10k steps, embedding2-only
      trainable, as the code was at the time). Skipped if the file is missing.

All are evaluated on identical batches, so differences are paired.
"""

import os
import statistics

import torch

from _common import (
    NUM_SOFT_TOKENS, REPO_ROOT, generate_softprompt, get_device, get_tokenizer, kd_loss, load_all,
    receiver_forward, setup_stdout, take_batches,
)

NUM_BATCHES = 20
CKPT = os.path.join(REPO_ROOT, "checkpoints", "embedding2_step10000.pt")


@torch.no_grad()
def loss_with_prefix(prefix_fn, teacher, receiver, batches, device):
    out = []
    for b in batches:
        ids = b.to(device)
        teacher_logits = teacher(input_ids=ids).logits
        out.append(kd_loss(receiver_forward(receiver, prefix_fn(ids), ids), teacher_logits).item())
    return out


def main():
    setup_stdout()
    device = get_device()
    tok = get_tokenizer()
    encoder, embedding2, w1, teacher, receiver = load_all(device)
    batches = take_batches(tok, NUM_BATCHES)

    if tok.bos_token_id is not None:
        tid, which = tok.bos_token_id, "BOS"
    else:
        tid, which = tok.pad_token_id, "pad"
    print(f"tokenizer bos_token={tok.bos_token!r} pad_token={tok.pad_token!r} eos_token={tok.eos_token!r}")
    print(f"(a) uses the {which} token: id={tid} {tok.convert_ids_to_tokens(tid)!r}"
          f"{'  (same id as EOS, the document separator in the data stream)' if tid == tok.eos_token_id else ''}")
    tok_vec = receiver.model.embed_tokens.weight[tid]
    hidden = tok_vec.shape[0]

    results = {}
    results[f"(a) {which} x100"] = loss_with_prefix(
        lambda ids: tok_vec.expand(ids.shape[0], NUM_SOFT_TOKENS, hidden), teacher, receiver, batches, device)
    results["(b) zeros x100"] = loss_with_prefix(
        lambda ids: torch.zeros(ids.shape[0], NUM_SOFT_TOKENS, hidden, device=device), teacher, receiver,
        batches, device)
    results["(c) generated, untrained"] = loss_with_prefix(
        lambda ids: generate_softprompt(encoder, embedding2, ids, NUM_SOFT_TOKENS), teacher, receiver,
        batches, device)
    if os.path.exists(CKPT):
        ckpt = torch.load(CKPT, map_location=device)
        print(f"checkpoint keys: {sorted(ckpt.keys())}  step={ckpt.get('step')}")
        with torch.no_grad():
            before = embedding2.weight.detach().clone()
            embedding2.load_state_dict(ckpt["embedding2_state_dict"])
            d = (embedding2.weight - before)
            rows_moved = (d.norm(dim=1) > 0).sum().item()
            print(f"step-10000 embedding2 vs init: ||dE||/||E||={(d.norm() / before.norm()).item():.4e}, "
                  f"max|dE|={d.abs().max().item():.4e}, rows changed={rows_moved}/{d.shape[0]}")
        results["(d) generated, emb2@step10000"] = loss_with_prefix(
            lambda ids: generate_softprompt(encoder, embedding2, ids, NUM_SOFT_TOKENS), teacher, receiver,
            batches, device)
    else:
        print(f"{CKPT} not found -- skipping (d)")

    print(f"\nper-batch KD loss ({NUM_BATCHES} batches x {batches[0].shape[0]} prompts x "
          f"{batches[0].shape[1]} tokens):")
    names = list(results)
    print(f"{'batch':>5s} " + " ".join(f"{n[:22]:>24s}" for n in names))
    for i in range(NUM_BATCHES):
        print(f"{i + 1:5d} " + " ".join(f"{results[n][i]:24.6f}" for n in names))

    ref = results["(c) generated, untrained"]
    print("\nSUMMARY (mean +- std over batches; paired diff vs (c) untrained generated)")
    for n in names:
        v = results[n]
        diffs = [x - r for x, r in zip(v, ref)]
        wins = sum(x < r for x, r in zip(v, ref))
        print(f"  {n:32s} mean={statistics.mean(v):.4f} +- {statistics.stdev(v):.4f}   "
              f"mean diff vs (c)={statistics.mean(diffs):+.4f}   lower than (c) on {wins}/{NUM_BATCHES} batches")


if __name__ == "__main__":
    main()
