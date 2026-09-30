"""Test 4 - naive prefixes vs. gists on the 32 held-out prompts (training KD loss).

Prefixes: text16, zeros, pad16, rand_tok (seeds 0/1/2), and the step0 / step1000 / best gists.
Mean and all 8 per-batch values (batches of 4, as in the training eval).
"""

import statistics

import torch

from _gist_common import Setup, save_json


def main():
    s = Setup()
    refs = s.reference_prefixes()
    teacher = [s.teacher_logits(b.to(s.device)) for b in s.heldout]
    rows = {}

    def score(name, prefix_fn):
        per_batch = []
        for b, t in zip(s.heldout, teacher):
            ids = b.to(s.device)
            per_batch.append(statistics.mean(s.per_prompt_kd(prefix_fn(ids), ids, t)))
        rows[name] = {"mean": statistics.mean(per_batch), "per_batch": per_batch}
        print(f"{name:12s} mean={rows[name]['mean']:.5f} per-batch={[round(x, 4) for x in per_batch]}")

    for rname, vec in refs.items():
        score(rname, lambda ids, v=vec: v.unsqueeze(0).expand(ids.shape[0], -1, -1))
    rt = [rows[f"rand_tok_s{k}"]["mean"] for k in (0, 1, 2)]
    rows["rand_tok_mean_of_3"] = {"mean": statistics.mean(rt), "per_draw": rt}
    for name in s.names:
        s.use(name)
        with torch.no_grad():
            score(f"gist_{name}", lambda ids: s.gists([ids]))
    save_json("t4_baselines.json", {"info": s.info(), "references": s.reference_info(), "results": rows})


if __name__ == "__main__":
    main()
