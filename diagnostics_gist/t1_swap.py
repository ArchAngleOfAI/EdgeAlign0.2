"""Test 1 - swap test: does prompt A's loss change when it gets prompt B's gist instead of its own?

32 held-out prompts; B = a fixed random derangement of A (no prompt keeps its own gist), seeds 0/1/2.
The teacher always sees plain A. Per-prompt KD loss is the training KD loss on A's prompt positions.
"""

import statistics

import torch

from _gist_common import Setup, batch_split, derangement, save_json

DERANGEMENT_SEEDS = (0, 1, 2)


def main():
    s = Setup()
    ids = torch.cat(s.heldout).to(s.device)            # (32, 128)
    teacher = torch.cat([s.teacher_logits(b) for b in batch_split(ids)])
    out = {"info": s.info(), "derangement_seeds": list(DERANGEMENT_SEEDS), "results": {}}
    for name in s.names:
        s.use(name)
        g = s.gists(s.heldout)                          # (32, 16, H)
        own = sum((s.per_prompt_kd(gb, ib, tb) for gb, ib, tb in
                   zip(batch_split(g), batch_split(ids), batch_split(teacher))), [])
        res = {"own_mean": statistics.mean(own), "own_per_prompt": own, "swaps": {}}
        for seed in DERANGEMENT_SEEDS:
            perm = derangement(ids.shape[0], seed)
            gs = g[perm]
            sw = sum((s.per_prompt_kd(gb, ib, tb) for gb, ib, tb in
                      zip(batch_split(gs), batch_split(ids), batch_split(teacher))), [])
            d = [a - b for a, b in zip(sw, own)]
            res["swaps"][seed] = {"perm": perm, "swapped_mean": statistics.mean(sw),
                                  "paired_diff_mean": statistics.mean(d), "paired_diff_std": statistics.stdev(d),
                                  "frac_swapped_gt_own": sum(x > 0 for x in d) / len(d),
                                  "swapped_per_prompt": sw}
            print(f"{name} seed{seed}: own={res['own_mean']:.5f} swapped={statistics.mean(sw):.5f} "
                  f"diff={statistics.mean(d):+.5f} frac>={res['swaps'][seed]['frac_swapped_gt_own']:.3f}")
        out["results"][name] = res
    save_json("t1_swap.json", out)


if __name__ == "__main__":
    main()
