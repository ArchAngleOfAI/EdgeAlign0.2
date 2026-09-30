"""Test 7 - L2 norms of the 16 gist vectors vs. the receiver's real token embeddings."""

import torch

from _gist_common import L, Setup, save_json


def main():
    s = Setup()
    en = s.receiver.model.embed_tokens.weight.detach().norm(dim=-1)
    n_regular = min(s.tok.all_special_ids)
    en_reg = en[:n_regular]
    ref = {"all_rows": {"mean": en.mean().item(), "p5": en.quantile(0.05).item(), "p95": en.quantile(0.95).item()},
           "ordinary_tokens": {"n": n_regular, "mean": en_reg.mean().item(), "p5": en_reg.quantile(0.05).item(),
                               "p95": en_reg.quantile(0.95).item()}}
    print(f"receiver embed_tokens norms: {ref}")
    res = {}
    for name in s.names:
        s.use(name)
        n = s.gists(s.heldout).norm(dim=-1)                      # (32, 16)
        res[name] = {"per_position": [{"pos": k + 1, "mean": n[:, k].mean().item(), "min": n[:, k].min().item(),
                                       "max": n[:, k].max().item()} for k in range(L)],
                     "overall": {"mean": n.mean().item(), "min": n.min().item(), "max": n.max().item()}}
        print(f"{name}: gist norms mean {n.mean().item():.3f} min {n.min().item():.3f} max {n.max().item():.3f}; "
              f"by position {[round(n[:, k].mean().item(), 2) for k in range(L)]}")
    save_json("t7_norms.json", {"info": s.info(), "embed_tokens_norms": ref, "results": res})


if __name__ == "__main__":
    main()
