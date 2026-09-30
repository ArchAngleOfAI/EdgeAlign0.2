"""Test 5 - how much does the receiver's output move when the prefix is perturbed?

For each prefix (step0 / step1000 / best gists, text16) and f in {0.01, 0.05, 0.1, 0.5}: add Gaussian
noise to every one of the 16 vectors with ||noise|| = f * ||vector||, and measure
KL(receiver | clean prefix  ||  receiver | noisy prefix) on the prompt positions after the prefix
(the training KD loss with the clean-prefix receiver in the teacher slot), averaged over the 32
held-out prompts. 3 noise seeds (0, 1, 2). Plot: KL vs f, log-log, one line per prefix.
"""

import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from _gist_common import RESULTS_DIR, Setup, batch_split, kd_loss, receiver_forward, save_json  # noqa: E402

FRACS = (0.01, 0.05, 0.1, 0.5)
NOISE_SEEDS = (0, 1, 2)
COLORS = {"gist_step0": "#2a78d6", "gist_step1000": "#eb6834", "gist_best": "#1baf7a", "text16": "#e87ba4"}


@torch.no_grad()
def noisy_kl(s, prefix, ids, f, seed):
    g = torch.Generator(device=s.device).manual_seed(seed)
    vals = []
    for pb, ib in zip(batch_split(prefix), batch_split(ids)):
        clean = receiver_forward(s.receiver, pb, ib)
        n = torch.randn(pb.shape, generator=g, device=s.device)
        n = n / n.norm(dim=-1, keepdim=True) * pb.norm(dim=-1, keepdim=True) * f
        noisy = receiver_forward(s.receiver, pb + n, ib)
        vals += [kd_loss(noisy[i:i + 1], clean[i:i + 1]).item() for i in range(ib.shape[0])]
    return statistics.mean(vals)


def main():
    s = Setup()
    ids = torch.cat(s.heldout).to(s.device)
    prefixes = {}
    for name in s.names:
        s.use(name)
        prefixes[f"gist_{name}"] = s.gists(s.heldout)
    prefixes["text16"] = s.reference_prefixes()["text16"].unsqueeze(0).expand(ids.shape[0], -1, -1)
    res = {}
    for pname, pre in prefixes.items():
        res[pname] = {}
        for f in FRACS:
            per_seed = [noisy_kl(s, pre, ids, f, seed) for seed in NOISE_SEEDS]
            res[pname][str(f)] = {"mean": statistics.mean(per_seed), "std": statistics.stdev(per_seed),
                                  "per_seed": per_seed}
            print(f"{pname:14s} f={f:<5} KL mean={statistics.mean(per_seed):.3e} std={statistics.stdev(per_seed):.1e}")

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    for pname, r in res.items():
        m = [r[str(f)]["mean"] for f in FRACS]
        sd = [r[str(f)]["std"] for f in FRACS]
        ax.errorbar(FRACS, m, yerr=sd, marker="o", ms=5, lw=2, capsize=3, label=pname,
                    color=COLORS.get(pname))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise fraction f  (||noise|| = f · ||prefix vector||)")
    ax.set_ylabel("KL(receiver | clean prefix || receiver | noisy prefix)")
    ax.set_title("Receiver sensitivity to prefix noise (32 held-out prompts, 3 noise seeds, L=16)")
    ax.grid(True, which="both", color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/sensitivity.png")
    save_json("t5_sensitivity.json", {"info": s.info(), "fracs": list(FRACS), "noise_seeds": list(NOISE_SEEDS),
                                      "results": res})


if __name__ == "__main__":
    main()
