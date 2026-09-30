"""Test 2 - how much do gists vary across prompts?

Gists for the 32 held-out + 256 extra held-out-shard prompts ([288, 16, H]). Per position k:
- mean pairwise cosine similarity between gists of DIFFERENT prompts,
- ||std over prompts|| / ||mean over prompts|| (std taken per hidden dim, then the vector's L2 norm).
Plus the mean cosine similarity between DIFFERENT positions within the same gist.
"""

import torch

from _gist_common import L, Setup, save_json


def offdiag_mean(sim):
    n = sim.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return sim[..., mask].mean(-1)


def main():
    s = Setup()
    batches = s.heldout + s.extra_batches()
    out = {"info": s.info(), "n_prompts": sum(b.shape[0] for b in batches), "results": {}}
    for name in s.names:
        s.use(name)
        g = s.gists(batches).double()                    # (N, 16, H)
        gn = torch.nn.functional.normalize(g, dim=-1)
        per_pos = []
        for k in range(L):
            v = gn[:, k, :]
            cross_cos = offdiag_mean(v @ v.T).item()
            std_ratio = (g[:, k, :].std(0).norm() / g[:, k, :].mean(0).norm()).item()
            per_pos.append({"pos": k + 1, "cross_prompt_cos": cross_cos, "std_over_mean_norm": std_ratio})
        within = offdiag_mean(gn @ gn.transpose(1, 2))    # (N,) mean cos between positions of one gist
        res = {"per_position": per_pos,
               "cross_prompt_cos_mean_over_positions": sum(p["cross_prompt_cos"] for p in per_pos) / L,
               "std_ratio_mean_over_positions": sum(p["std_over_mean_norm"] for p in per_pos) / L,
               "within_gist_cross_position_cos": {"mean": within.mean().item(), "min": within.min().item(),
                                                  "max": within.max().item()}}
        out["results"][name] = res
        print(f"{name}: cross-prompt cos (mean over pos) {res['cross_prompt_cos_mean_over_positions']:.4f}, "
              f"std/mean {res['std_ratio_mean_over_positions']:.4f}, "
              f"within-gist cos {res['within_gist_cross_position_cos']['mean']:.4f}")
    save_json("t2_variation.json", out)


if __name__ == "__main__":
    main()
