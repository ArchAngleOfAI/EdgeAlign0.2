"""Test 3 - inside the rollout: what distribution does the generator produce at each of the 16 steps?

generate_softprompt is called unchanged; a forward hook on the encoder's lm_head captures the logits
it computes at each rollout step (the first lm_head call per step is exactly softmax's input).
Reports per step: entropy (nats) mean/min/max over prompts, top-1 prob, the most common top-1 token;
per prompt: number of distinct top-1 tokens over the 16 steps; for 5 example prompts, the 5 nearest
RECEIVER embed_tokens rows (cosine) to each of the 16 soft vectors.
"""

import collections

import torch

from _gist_common import L, Setup, batch_split, generate_softprompt, save_json

EXAMPLES = 5


def main():
    s = Setup()
    ids = torch.cat(s.heldout).to(s.device)
    emb = torch.nn.functional.normalize(s.receiver.model.embed_tokens.weight.detach(), dim=-1)
    captured = []
    hook = s.encoder.lm_head.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    out = {"info": s.info(), "results": {}}
    for name in s.names:
        s.use(name)
        ents, top1p, top1 = [], [], []
        gist_all = []
        with torch.no_grad():
            for b in batch_split(ids):
                captured.clear()
                gist_all.append(generate_softprompt(s.encoder, s.embedding2, b, L))
                assert len(captured) == L, len(captured)
                logits = torch.cat(captured, 1).double()             # (B, 16, V)
                logp = torch.log_softmax(logits, -1)
                ents.append(-(logp.exp() * logp).sum(-1))            # (B, 16)
                p, t = logp.exp().max(-1)
                top1p.append(p)
                top1.append(t)
        ent, p1, t1 = torch.cat(ents), torch.cat(top1p), torch.cat(top1)   # (32, 16)
        gist = torch.cat(gist_all)
        per_step = []
        for k in range(L):
            c = collections.Counter(t1[:, k].tolist()).most_common(3)
            per_step.append({"step": k + 1, "entropy_mean": ent[:, k].mean().item(),
                             "entropy_min": ent[:, k].min().item(), "entropy_max": ent[:, k].max().item(),
                             "top1_prob_mean": p1[:, k].mean().item(), "top1_prob_min": p1[:, k].min().item(),
                             "top1_prob_max": p1[:, k].max().item(),
                             "top1_most_common": [{"token": s.tok.decode([tid]), "id": tid, "n_prompts": n}
                                                  for tid, n in c]})
        distinct = [len(set(t1[i].tolist())) for i in range(t1.shape[0])]
        examples = []
        for i in range(EXAMPLES):
            sims = torch.nn.functional.normalize(gist[i], dim=-1) @ emb.T     # (16, V)
            top = sims.topk(5, dim=-1)
            examples.append({"prompt_index": i,
                             "prompt_start": s.tok.decode(ids[i, :24].tolist()),
                             "top1_tokens_per_step": [s.tok.decode([x]) for x in t1[i].tolist()],
                             "nearest_receiver_tokens": [
                                 [{"token": s.tok.decode([tid]), "cos": round(c, 4)}
                                  for tid, c in zip(top.indices[k].tolist(), top.values[k].tolist())]
                                 for k in range(L)]})
        out["results"][name] = {"per_step": per_step, "distinct_top1_per_prompt": distinct,
                                "distinct_top1_mean": sum(distinct) / len(distinct), "examples": examples}
        print(f"{name}: entropy by step {[round(x['entropy_mean'], 2) for x in per_step]}")
        print(f"   top1 prob by step {[round(x['top1_prob_mean'], 3) for x in per_step]}")
        print(f"   distinct top-1 per prompt: mean {sum(distinct) / len(distinct):.2f} "
              f"min {min(distinct)} max {max(distinct)}")
    hook.remove()
    save_json("t3_rollout.json", out)


if __name__ == "__main__":
    main()
