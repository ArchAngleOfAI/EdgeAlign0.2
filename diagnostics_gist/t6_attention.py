"""Test 6 - how much attention does the receiver put on the prefix?

A SEPARATE copy of the receiver is loaded with attn_implementation="eager" (same pretrained weights,
frozen) so attention weights can be returned. Per layer, averaged over heads, over query positions
after the prefix (the 128 prompt positions) and over the 32 held-out prompts:
- fraction of attention mass on the 16 prefix positions,
- fraction on prefix position 0 alone.
Prefixes: step0 / step1000 / best gists, text16, zeros. The input is built as receiver_forward builds
it (torch.cat([prefix, embed_tokens(prompt)])); the eager copy's logits are checked against the
normal receiver_forward output.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from _gist_common import (  # noqa: E402
    L, MODEL_PATH, RESULTS_DIR, Setup, batch_split, receiver_forward, save_json,
)

COLORS = {"gist_step0": "#2a78d6", "gist_step1000": "#eb6834", "gist_best": "#1baf7a",
          "text16": "#e87ba4", "zeros": "#898781"}


@torch.no_grad()
def attention_fracs(s, eager, prefix, ids):
    tot_prefix, tot_pos0, n, max_logit_diff = None, None, 0, 0.0
    for pb, ib in zip(batch_split(prefix), batch_split(ids)):
        combined = torch.cat([pb, eager.model.embed_tokens(ib)], dim=1)
        out = eager.model(inputs_embeds=combined, output_attentions=True)
        eager_logits = eager.lm_head(out.last_hidden_state)[:, L:, :]
        max_logit_diff = max(max_logit_diff,
                             (eager_logits - receiver_forward(s.receiver, pb, ib)).abs().max().item())
        # attentions: tuple(layers) of (B, heads, T, T); queries after the prefix
        fp = torch.stack([a[:, :, L:, :L].sum(-1).mean(dim=(1, 2)) for a in out.attentions], 1)  # (B, layers)
        f0 = torch.stack([a[:, :, L:, 0].mean(dim=(1, 2)) for a in out.attentions], 1)
        tot_prefix = fp.sum(0) if tot_prefix is None else tot_prefix + fp.sum(0)
        tot_pos0 = f0.sum(0) if tot_pos0 is None else tot_pos0 + f0.sum(0)
        n += ib.shape[0]
    return (tot_prefix / n).tolist(), (tot_pos0 / n).tolist(), max_logit_diff


def main():
    s = Setup()
    eager = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32,
                                                 attn_implementation="eager").to(s.device)
    eager.eval()
    for p in eager.parameters():
        p.requires_grad = False
    same = all(torch.equal(a, b) for a, b in zip(eager.state_dict().values(), s.receiver.state_dict().values()))
    print(f"eager receiver weights identical to receiver: {same}")
    ids = torch.cat(s.heldout).to(s.device)
    prefixes = {}
    for name in s.names:
        s.use(name)
        prefixes[f"gist_{name}"] = s.gists(s.heldout)
    refs = s.reference_prefixes()
    for r in ("text16", "zeros"):
        prefixes[r] = refs[r].unsqueeze(0).expand(ids.shape[0], -1, -1)
    res = {}
    for pname, pre in prefixes.items():
        fp, f0, diff = attention_fracs(s, eager, pre, ids)
        res[pname] = {"prefix_frac_per_layer": fp, "pos0_frac_per_layer": f0,
                      "prefix_frac_mean_over_layers": sum(fp) / len(fp),
                      "pos0_frac_mean_over_layers": sum(f0) / len(f0),
                      "max_abs_logit_diff_eager_vs_receiver_forward": diff}
        print(f"{pname:14s} prefix frac (mean over layers) {res[pname]['prefix_frac_mean_over_layers']:.4f} "
              f"pos0 {res[pname]['pos0_frac_mean_over_layers']:.4f}  eager-vs-sdpa max|dlogit| {diff:.2e}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    for pname, r in res.items():
        ax.plot(range(len(r["prefix_frac_per_layer"])), r["prefix_frac_per_layer"], marker="o", ms=4, lw=2,
                label=pname, color=COLORS.get(pname))
    ax.set_xlabel("receiver layer")
    ax.set_ylabel("attention mass on the 16 prefix positions")
    ax.set_title("Receiver attention on the prefix (queries = prompt positions; mean over heads, 32 prompts)")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/attention.png")
    save_json("t6_attention.json", {"info": s.info(), "eager_weights_identical": same, "results": res})


if __name__ == "__main__":
    main()
