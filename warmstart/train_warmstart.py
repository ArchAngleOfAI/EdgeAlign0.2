"""Warm-start gist pretraining: "fill in the removed part".

Per example: the frozen teacher sees the full user message + response; the generator reads only
the removed part and rolls out L=16 gist vectors; the frozen receiver sees the user message with the
removed part replaced by the gist (no-gap examples: removed part kept, gist after it) + the same
response. Loss = KL(teacher || receiver), T=1, on the response tokens only (assistant content + its
closing <|im_end|>), token-weighted mean over all response tokens of the optimizer step's 16 examples
(changed 2026-10-01 from a token mean per micro-batch averaged over micro-batches, so that the
objective does not depend on the micro-batch size).

Settings: generator = build_softprompt_generator with the last 3 layers + LM_head + Embedding2
trainable (via diagnostics/_common.load_all, as in train_3layer.py), full backprop through the
rollout; fp32; seed 0; AdamW betas (0.9, 0.95), eps 1e-6, weight decay 0.01 (AdamW's default, which
train_3layer.py used); LR linear warmup 0 -> 1e-4 over 100 steps then cosine to 1e-5 at the final
step; global batch 16 = micro x accum (default 2 x 8; --micro-batch 4 -> 4 x 4 with the batched rollout); clip_grad_norm_ 1.0 once per step;
5,000 steps. Data order: a fresh permutation per epoch (seed 0 + epoch), no example repeated within
an epoch (the user chose multi-epoch because the 50/30/20 set is smaller than 80,000 draws).

Checkpoints/logs/plots: see RUN_NOTES_warmstart.md. --smoke writes everything to scratch locations.
"""

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # lets check (e1) switch on deterministic algorithms
import torch
import torch.nn.functional as F

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
from common import (  # noqa: E402
    DATA_DIR, L, PAD_ID, REPO_ROOT, LogitCapture, build_batch, kd_per_token, per_example_kl, receiver_inputs,
    receiver_response_logits, rollout_gists, rollout_gists_serial, split_per_example, teacher_response_logits,
)
from _common import get_device, load_all, setup_stdout, trainable_params  # noqa: E402
from train_3layer import (  # noqa: E402
    atomic_save, link_or_replace, load_trainable_state, moving_average, trainable_state, truncate_jsonl,
)

NUM_LAYERS = 3
PEAK_LR, MIN_LR = 1e-4, 1e-5
WARMUP_STEPS = 100
BETAS, EPS, WEIGHT_DECAY = (0.9, 0.95), 1e-6, 0.01
MICRO_BATCH, ACCUM = 2, 8
GLOBAL_BATCH = MICRO_BATCH * ACCUM
MAX_NORM = 1.0
TOTAL_STEPS = 5000
PERMANENT_EVERY = 1000
SEED = 0
ENTROPY_WARN = 0.05
EVAL_BS = 8
ATTN_N = 32
DEFAULT_CKPT_DIR = "/data/a84460786/testfolder_checkpoints/warmstart"

COL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
       "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
COLOR_MUTED, COLOR_GRID, COLOR_SURFACE, COLOR_TEXT = "#898781", "#e1e0d9", "#fcfcfb", "#3d3c38"


def lr_at(step, total):
    """LR for optimizer step `step` (1-indexed)."""
    if step <= WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    frac = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * frac))


def is_eval_step(step):
    return step == 0 or (step <= 300 and step % 10 == 0) or step % 100 == 0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class Order:
    """Example index for global draw g: permutation (seed SEED + epoch) of the training set."""

    def __init__(self, n):
        self.n, self.cache = n, {}

    def __call__(self, g):
        ep, pos = divmod(g, self.n)
        if ep not in self.cache:
            perm = list(range(self.n))
            random.Random(SEED + ep).shuffle(perm)
            self.cache = {ep: perm}
        return self.cache[ep][pos], ep


def bucket(n_removed, cap):
    return "le64" if n_removed <= 64 else ("65_128" if n_removed <= 128 else f"129_{cap}")


def derangement(n, seed=0):
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != j for i, j in enumerate(p)):
            return p


# ---------------------------------------------------------------- forward helpers

def micro_loss(models, exs, capture, device, entropy_coef, n_tok=None, n_micro=1):
    """KD = this micro-batch's share of the token-weighted mean over the whole optimizer step: summed per-token KL
    divided by n_tok, the step's total response tokens (default: this micro-batch's own, i.e. its token mean).
    Summing the shares over a step's micro-batches gives one token-weighted mean over all 16 examples, whatever
    the micro-batch size. The entropy term stays a per-gist-position mean, split evenly over n_micro micro-batches."""
    encoder, embedding2, teacher, receiver = models
    gists, ent, _ = rollout_gists(encoder, embedding2, [e["removed_ids"] for e in exs], capture, device)
    b = build_batch(exs, device)
    t_log = teacher_response_logits(teacher, b)
    r_log = receiver_response_logits(receiver, b, list(gists))
    kd = kd_per_token(r_log, t_log).sum() / (n_tok or sum(b["resp_len"]))
    mean_ent = ent.mean()
    return kd - entropy_coef * mean_ent / n_micro, kd, mean_ent, gists


@torch.no_grad()
def kl_with_middles(teacher, receiver, exs, middles, device):
    """Per-example response KL for each list of middles (same teacher logits reused)."""
    out = [[] for _ in middles]
    for i in range(0, len(exs), EVAL_BS):
        chunk = exs[i:i + EVAL_BS]
        b = build_batch(chunk, device)
        t_log = teacher_response_logits(teacher, b)
        for k, mids in enumerate(middles):
            r_log = receiver_response_logits(receiver, b, mids[i:i + EVAL_BS])
            out[k] += per_example_kl(kd_per_token(r_log, t_log), b["resp_len"])
            del r_log
        del t_log
    return out


@torch.no_grad()
def kl_nogist(teacher, receiver, exs, device):
    H = receiver.config.hidden_size
    empty = [torch.zeros((0, H), device=device) for _ in exs]
    gap_exs = [dict(e, gap=True) for e in exs]
    return kl_with_middles(teacher, receiver, gap_exs, [empty], device)[0]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows):
    """rows: per-example dicts. Metrics on gap / no-gap subsets."""
    gap = [r for r in rows if r["gap"]]
    nog = [r for r in rows if not r["gap"]]
    s = {"n": len(rows), "n_gap": len(gap), "n_nogap": len(nog),
         "kl_gist_gap": mean(r["kl_gist"] for r in gap), "kl_gist_nogap": mean(r["kl_gist"] for r in nog),
         "entropy": mean(r["entropy"] for r in rows), "distinct_top1": mean(r["distinct_top1"] for r in rows)}
    if gap:
        mn = mean(r["kl_nogist"] for r in gap)
        s["kl_nogist_gap"] = mn
        s["recovery_mean_ratio"] = mean((r["kl_nogist"] - r["kl_gist"]) / r["kl_nogist"] for r in gap)
        s["recovery_ratio_of_means"] = (mn - s["kl_gist_gap"]) / mn
        if "kl_swap" in gap[0]:
            s["swap_minus_own_gap"] = mean(r["kl_swap"] - r["kl_gist"] for r in gap)
            s["swap_worse_frac_gap"] = mean(float(r["kl_swap"] > r["kl_gist"]) for r in gap)
    return s


def pairwise_cos(gists):
    """Mean cosine between gists of different examples: (a) flattened 16xH, (b) per position then averaged."""
    n = gists.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=gists.device)
    flat = F.normalize(gists.reshape(n, -1), dim=-1)
    c_flat = (flat @ flat.T)[off].mean().item()
    per = F.normalize(gists, dim=-1).transpose(0, 1)  # (L, n, H)
    c_pos = torch.stack([(p @ p.T)[off].mean() for p in per]).mean().item()
    return c_flat, c_pos


@torch.no_grad()
def attention_fraction(receiver_eager, exs, gists, device):
    """Share of attention from response query positions onto gist positions 1..16 and 2..16,
    averaged over heads, layers, response queries and examples (eager-attention receiver copy)."""
    f_all, f_2 = [], []
    for e, g in zip(exs, gists):
        b = build_batch([e], device)
        embeds, mask, starts = receiver_inputs(receiver_eager, b, [g])
        out = receiver_eager.model(inputs_embeds=embeds, attention_mask=mask, output_attentions=True)
        g0 = len(b["before"][0])
        q0, nq = starts[0], b["resp_len"][0]
        a_all, a_2 = [], []
        for att in out.attentions:  # (1, heads, T, T)
            q = att[0, :, q0:q0 + nq, :]
            a_all.append(q[:, :, g0:g0 + L].sum(-1).mean())
            a_2.append(q[:, :, g0 + 1:g0 + L].sum(-1).mean())
        f_all.append(torch.stack(a_all).mean().item())
        f_2.append(torch.stack(a_2).mean().item())
        del out
    return mean(f_all), mean(f_2)


# ---------------------------------------------------------------- plots

def _style(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def _save(fig, path):
    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=COLOR_SURFACE)
    os.replace(tmp, path)


def save_plots(train_rows, eval_rows, nogist_mean, out_dir, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style(ax)
    if train_rows:
        s = [r["step"] for r in train_rows]
        v = [r["train_kd"] for r in train_rows]
        ax.plot(s, v, color=COL["blue"], lw=0.7, alpha=0.3, label="train KD loss (per step, 16 examples)")
        ax.plot(s, moving_average(v), color=COL["blue"], lw=2, label="train KD loss, 50-step moving average")
    if eval_rows:
        ax.plot([r["step"] for r in eval_rows], [r["A"]["overall"]["kl_gist_gap"] for r in eval_rows],
                color=COL["orange"], lw=2, marker="o", ms=4, markeredgecolor=COLOR_SURFACE,
                label="held-out A KL_gist (gap examples)")
    ax.axhline(nogist_mean, color=COLOR_MUTED, lw=1.5, ls=":",
               label=f"held-out A mean KL_nogist (gap) = {nogist_mean:.3f}")
    ax.axvline(WARMUP_STEPS, color=COLOR_MUTED, ls="--", lw=1.2, label="end of LR warmup (step 100)")
    ax.set_xlim(0, max(total, 1))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("optimizer step", color=COLOR_TEXT)
    ax.set_ylabel("KL(teacher || receiver) on response tokens, T=1", color=COLOR_TEXT)
    ax.set_title("Warm-start gist pretraining (fill in the removed part), L=16\n"
                 "AdamW peak 1e-4, warmup 100 + cosine to 1e-5, global batch 16, clip 1.0", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "warmstart_loss_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=130)
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style(ax)
    if eval_rows:
        s = [r["step"] for r in eval_rows]
        ax.plot(s, [r["A"]["overall"]["recovery_mean_ratio"] for r in eval_rows], color=COL["blue"], lw=2.2, marker="o", ms=3,
                label="held-out A recovery, overall (mean of per-example ratios)")
        for src, c in (("sni", COL["orange"]), ("rlvr", COL["aqua"]), ("squad", COL["yellow"])):
            ax.plot(s, [r["A"]["by_source"][src].get("recovery_mean_ratio", float("nan")) for r in eval_rows],
                    color=c, lw=1.3, marker="o", ms=2, label=f"held-out A recovery, {src}")
        sb = [(r["step"], r["B"]["overall"]["recovery_mean_ratio"]) for r in eval_rows if r.get("B")]
        if sb:
            ax.plot(*zip(*sb), color=COL["magenta"], lw=2, marker="s", ms=5, ls="--",
                    label="held-out B recovery (unseen SNI tasks)")
        ax.plot(s, [r["A"]["overall"]["swap_minus_own_gap"] for r in eval_rows], color=COL["green"], lw=1.3, marker="o", ms=2,
                label="held-out A swap − own KL (gap)")
        ax.plot(s, [r["A"]["cross_cos_flat"] for r in eval_rows], color=COL["violet"], lw=1.3, marker="o", ms=2,
                label="held-out A cross-example gist cosine")
        ax2 = ax.twinx()
        ax2.plot(s, [r["A"]["overall"]["entropy"] for r in eval_rows], color=COL["red"], lw=1.3, ls=":", marker="o", ms=2,
                 label="held-out A rollout entropy (nats, right axis)")
        ax2.set_ylabel("mean rollout softmax entropy (nats)", color=COLOR_TEXT)
        ax2.set_ylim(bottom=0)
        ax2.tick_params(colors=COLOR_MUTED)
        for side in ("top",):
            ax2.spines[side].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)
    ax.axhline(0, color=COLOR_MUTED, lw=0.8)
    ax.axvline(WARMUP_STEPS, color=COLOR_MUTED, ls="--", lw=1.2)
    ax.set_xlim(0, max(total, 1))
    ax.set_xlabel("optimizer step", color=COLOR_TEXT)
    ax.set_ylabel("recovery / KL difference / cosine", color=COLOR_TEXT)
    ax.set_title("Gist usefulness and gist diversity (dashed line: end of warmup, step 100)", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "warmstart_recovery.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Step 2 checks

def run_checks(models, train, hold_a, hold_b, capture, device, params):
    """(a)-(e); returns a dict and raises SystemExit on failure."""
    encoder, embedding2, teacher, receiver = models
    res = {}
    # (a) response ids identical in teacher and receiver sequences, for every example
    bad = 0
    for e in train + hold_a + hold_b:
        t_seq = e["prefix_ids"] + e["removed_ids"] + e["suffix_ids"] + e["response_ids"]
        after = e["suffix_ids"] + e["response_ids"]
        n = len(e["response_ids"])
        if t_seq[-n:] != e["response_ids"] or after[-n:] != e["response_ids"] or e["response_ids"][-1] != 151645:
            bad += 1
    res["a_response_ids_identical"] = {"examples": len(train) + len(hold_a) + len(hold_b), "mismatches": bad}
    if bad:
        sys.exit(f"CHECK (a) FAILED: {bad} mismatches")

    # (b) alignment: removed part exactly 16 tokens, real embeddings in place of the gist -> KL ~ 0
    ex16 = [e for e in train + hold_a if len(e["removed_ids"]) == L and e["gap"]][:64]
    emb = receiver.model.embed_tokens
    mids = [emb(torch.tensor(e["removed_ids"], device=device)) for e in ex16]
    kl16 = kl_with_middles(teacher, receiver, ex16, [mids], device)[0]
    res["b_alignment_kl"] = {"n_examples": len(ex16), "mean": mean(kl16), "max": max(kl16)}
    if not ex16 or max(kl16) > 1e-3:
        sys.exit(f"CHECK (b) FAILED: {res['b_alignment_kl']}")

    # (c) no gist on gap examples -> KL clearly > 0
    gap64 = [e for e in hold_a if e["gap"]][:64]
    kl_ng = kl_nogist(teacher, receiver, gap64, device)
    res["c_nogist_kl"] = {"n_examples": len(gap64), "mean": mean(kl_ng), "min": min(kl_ng)}
    if mean(kl_ng) < 0.05:
        sys.exit(f"CHECK (c) FAILED: {res['c_nogist_kl']}")

    # (d) one backward pass: nonzero grads on embedding2, lm_head, the 3 layers; grad_fn on last soft token
    for p in params:
        p.grad = None
    exs = train[:MICRO_BATCH]
    loss, kd, ent, gists = micro_loss(models, exs, capture, device, 0.0)
    last_fn = gists[:, -1, :].grad_fn
    loss.backward()
    norms = {"embedding2": embedding2.weight.grad.norm().item(), "lm_head": encoder.lm_head.weight.grad.norm().item()}
    for k, layer in enumerate(encoder.model.layers[-NUM_LAYERS:]):
        norms[f"layer_{len(encoder.model.layers) - NUM_LAYERS + k}"] = math.sqrt(
            sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None))
    frozen_grad = [n for n, p in teacher.named_parameters() if p.grad is not None] + \
                  [n for n, p in receiver.named_parameters() if p.grad is not None] + \
                  ([] if encoder.model.embed_tokens.weight.grad is None else ["encoder.embed_tokens"])
    res["d_backward"] = {"loss": kd.item(), "grad_norms": norms, "last_soft_token_grad_fn": str(last_fn),
                         "frozen_params_with_grad": frozen_grad}
    for p in params:
        p.grad = None
    if last_fn is None or any(not (v > 0 and math.isfinite(v)) for v in norms.values()) or frozen_grad:
        sys.exit(f"CHECK (d) FAILED: {res['d_backward']}")

    # (e) batched (left-padded) rollout. It cannot match the per-example rollout exactly: batch shape alone changes
    # fp32 reductions (~1e-6; Qwen3RMSNorm runs in fp32 even for an fp64 model), and this rollout amplifies that to
    # percent level in gists and more in grads (the per-example rollout is no closer to an fp64 reference). So test
    # what padding can break: (e1) a padding leak -> swap the pad token at identical shapes: gists, entropy and grads
    # must be bit-identical (under deterministic algorithms); (e2) position ids -> first soft token vs an unpadded batch of copies, at fp32 noise level.
    by_len = sorted(train[:256], key=lambda e: len(e["removed_ids"]))
    removed = [by_len[round(k * (len(by_len) - 1) / 3)]["removed_ids"] for k in range(4)]  # spans the length range
    proj = torch.randn((len(removed), L, encoder.config.hidden_size), generator=torch.Generator().manual_seed(0)).to(device)

    def run(batch, pad_id):
        for p in params:
            p.grad = None
        g, en, tp = rollout_gists(encoder, embedding2, batch, capture, device, pad_id=pad_id)
        ((g * proj[:len(batch)]).sum() + en.sum()).backward()
        out = g.detach(), en.detach(), tp, [p.grad.clone() for p in params]
        for p in params:
            p.grad = None
        return out

    torch.use_deterministic_algorithms(True)  # backward is otherwise not bit-reproducible (~4e-6 run to run)
    try:
        (g1, e1, t1, gr1), (g2, e2, t2, gr2) = run(removed, PAD_ID), run(removed, 0)
    finally:
        torch.use_deterministic_algorithms(False)
    leak = {"gists_bit_identical": bool(torch.equal(g1, g2)), "entropy_bit_identical": bool(torch.equal(e1, e2)),
            "top1_identical": bool(torch.equal(t1, t2)),
            "grads_bit_identical": all(torch.equal(a, b) for a, b in zip(gr1, gr2))}
    with torch.no_grad():
        first = []
        for i, r in enumerate(removed):
            gu, _, _ = rollout_gists(encoder, embedding2, [r] * len(removed), capture, device)
            first.append(((g1[i, 0] - gu[0, 0]).norm() / gu[0, 0].norm()).item())
    res["e_batched_rollout"] = {"removed_lens": [len(r) for r in removed], "e1_pad_token_swap": leak,
                                "e2_first_soft_token_rel_diff_vs_unpadded": first}
    if not all(leak.values()) or max(first) > 1e-4:
        sys.exit(f"CHECK (e) FAILED: {res['e_batched_rollout']}")
    print("STEP 2 CHECKS PASSED:", json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- main

def main():
    global MICRO_BATCH, ACCUM
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--schedule-steps", type=int, default=None,
                    help="Length of the LR schedule (default: --steps). A smoke run uses the full "
                         "5,000-step schedule so its LRs match the real run's first steps.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checks-only", action="store_true", help="Run the Step 2 checks, write them, exit.")
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH,
                    help="Examples per micro-batch; accum = 16 / this, so the global batch stays 16.")
    ap.add_argument("--skip-evals", action="store_true", help="Smoke timing: only the step-0 eval.")
    args = ap.parse_args()
    # Data parallel when launched with torchrun (one process per GPU): each rank takes an equal share of the
    # step's 16 examples, gradients are summed across ranks before clipping, so the update is the same as on
    # one GPU. Rank 0 alone runs the checks, evals, logging, plots and checkpoints.
    dist_on = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if dist_on:
        import datetime
        import torch.distributed as dist
        # GPU-to-GPU P2P hangs on this machine (NCCL broadcast never completes, 2026-10-01); route through host
        # memory instead (~3.5 GB/s between GPUs 4 and 6, ~0.4 s for the full gradient all-reduce)
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(minutes=30))
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        device, rank, world = get_device(), 0, 1
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")
    if GLOBAL_BATCH % (args.micro_batch * world):
        sys.exit(f"--micro-batch x {world} GPU(s) must divide {GLOBAL_BATCH}")
    MICRO_BATCH, ACCUM = args.micro_batch, GLOBAL_BATCH // (args.micro_batch * world)  # ACCUM is per rank
    CONFIG.update(micro_batch=MICRO_BATCH, accum=ACCUM, world_size=world)

    def on_main(fn, *a):
        """Run fn on rank 0 only while the other ranks wait; a STOP (SystemExit) on rank 0 stops every rank."""
        msg = None
        if is_main:
            try:
                fn(*a)
            except SystemExit as ex:
                msg = str(ex.code)
        if dist_on:
            flag = torch.tensor([0 if msg is None else 1], device=device)
            dist.broadcast(flag, 0)
            if flag.item():
                sys.exit(msg if is_main else "STOP (raised on rank 0)")
        elif msg is not None:
            sys.exit(msg)

    def check_sync(step):
        """Every rank must hold the same trainable weights (identical summed grads -> identical AdamW steps)."""
        if not dist_on:
            return
        mine = torch.stack([p.detach().double().sum() for p in params])
        allv = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(allv, mine)
        if not all(torch.equal(allv[0], v) for v in allv[1:]):
            sys.exit(f"STOP: trainable weights differ across GPUs at step {step}")
    sched = args.schedule_steps or args.steps

    if args.smoke:
        out_dir = os.path.join(REPO_ROOT, "logs", "smoke_warmstart")
        ckpt_dir = args.ckpt_dir.rstrip("/") + "_smoke"
    else:
        out_dir = REPO_ROOT
        ckpt_dir = args.ckpt_dir
    log_dir = out_dir if args.smoke else os.path.join(REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_log = os.path.join(log_dir, "warmstart_train.jsonl")
    eval_log = os.path.join(log_dir, "warmstart_eval.jsonl")
    latest = os.path.join(ckpt_dir, "latest.pt")

    torch.manual_seed(SEED)
    encoder, embedding2, _, teacher, receiver = load_all(device, num_trainable_layers=NUM_LAYERS)
    encoder.config.tie_word_embeddings = False
    ptrs = {encoder.lm_head.weight.data_ptr(), encoder.model.embed_tokens.weight.data_ptr(),
            embedding2.weight.data_ptr()}
    assert len(ptrs) == 3, "lm_head / embed_tokens / embedding2 must be three distinct tensors"
    params = trainable_params(encoder, embedding2, NUM_LAYERS)
    ids = {id(p) for p in params}
    assert all(p.requires_grad == (id(p) in ids) for p in encoder.parameters())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not any(p.requires_grad for p in receiver.parameters())
    print(f"model: {encoder.config._name_or_path}; trainable = embedding2 + lm_head + last {NUM_LAYERS} layers = "
          f"{sum(p.numel() for p in params):,} params; distinct data_ptrs OK")
    models = (encoder, embedding2, teacher, receiver)
    capture = LogitCapture(encoder.lm_head)
    optimizer = torch.optim.AdamW(params, lr=lr_at(1, sched), betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    hold_a = load_jsonl(os.path.join(DATA_DIR, "heldout_a.jsonl"))
    hold_b = load_jsonl(os.path.join(DATA_DIR, "heldout_b.jsonl"))
    stats = json.load(open(os.path.join(DATA_DIR, "data_stats.json")))
    cap = stats["sni"]["CAP"]
    print(f"data: train {len(train)}, held-out A {len(hold_a)}, held-out B {len(hold_b)}, CAP {cap}")

    def checks_main():
        checks = run_checks(models, train, hold_a, hold_b, capture, device, params)
        if args.checks_only:
            json.dump(checks, open(os.path.join(WS, "step2_checks.json"), "w"), indent=1)

    on_main(checks_main)
    if args.checks_only:
        if dist_on:
            dist.destroy_process_group()
        return

    # KL_nogist once at start for every gap example of both held-out sets (compared to the data file); evals
    # run on rank 0 only, so only rank 0 needs it
    def kl_nogist_main():
        t0 = time.time()
        for name, hs in (("A", hold_a), ("B", hold_b)):
            gap = [e for e in hs if e["gap"]]
            vals = kl_nogist(teacher, receiver, gap, device)
            diffs = [abs(v - e["kl_nogist"]) for v, e in zip(vals, gap)]
            for v, e in zip(vals, gap):
                e["kl_nogist"] = v
            print(f"KL_nogist held-out {name}: {len(gap)} gap examples, mean {mean(vals):.4f}; "
                  f"max |diff| vs data-prep value {max(diffs):.2e}")
        print(f"KL_nogist computed in {time.time() - t0:.0f}s")

    on_main(kl_nogist_main)
    nogist_mean_a = mean(e["kl_nogist"] for e in hold_a if e["gap"])
    der_a, der_b = derangement(len(hold_a), 0), derangement(len(hold_b), 0)
    attn_idx = [i for i, e in enumerate(hold_a) if e["gap"]][:ATTN_N]

    start_step, micro_consumed = 0, 0
    best = {"recovery": -math.inf, "step": None}
    train_rows, eval_rows = [], []
    if args.resume:
        ckpt = torch.load(latest, map_location=device)
        if ckpt["config"].get("micro_batch", 2) != MICRO_BATCH:  # micro_consumed counts micro-batches
            sys.exit(f"--resume needs --micro-batch {ckpt['config'].get('micro_batch', 2)} (the checkpoint's)")
        load_trainable_state(encoder, embedding2, ckpt["trainable"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, micro_consumed, best = ckpt["step"], ckpt["micro_consumed"], ckpt["best"]
        if is_main:
            train_rows = truncate_jsonl(train_log, start_step)
            eval_rows = truncate_jsonl(eval_log, start_step)
        print(f"resumed from {latest}: step {start_step}, best recovery {best['recovery']:.4f} @ {best['step']}")
    elif is_main:
        for p in (train_log, eval_log):
            if os.path.exists(p):
                os.remove(p)
    order = Order(len(train))

    @torch.no_grad()
    def eval_set(hs, der, do_swap=True):
        gists, ent, top = [], [], []
        for i in range(0, len(hs), EVAL_BS):
            g, en, tp = rollout_gists(encoder, embedding2, [e["removed_ids"] for e in hs[i:i + EVAL_BS]],
                                      capture, device)
            gists.append(g)
            ent.append(en)
            top.append(tp)
        gists, ent, top = torch.cat(gists), torch.cat(ent), torch.cat(top)
        mids = [list(gists)]
        if do_swap:
            mids.append([gists[der[i]] for i in range(len(hs))])
        kls = kl_with_middles(teacher, receiver, hs, mids, device)
        rows = []
        for i, e in enumerate(hs):
            r = {"id": e["id"], "source": e["source"], "gap": e["gap"], "n_removed": len(e["removed_ids"]),
                 "kl_gist": kls[0][i], "kl_nogist": e.get("kl_nogist") if e["gap"] else None,
                 "entropy": ent[i].mean().item(), "distinct_top1": len(set(top[i].tolist()))}
            if do_swap:
                r["kl_swap"] = kls[1][i]
            rows.append(r)
        return rows, gists

    def evaluate(step):
        t_ev = time.time()
        rows_a, gists_a = eval_set(hold_a, der_a)
        c_flat, c_pos = pairwise_cos(gists_a)
        res = {"step": step, "time": time.time(),
               "A": {"overall": summarize(rows_a),
                     "by_source": {s: summarize([r for r in rows_a if r["source"] == s]) for s in ("sni", "rlvr", "squad")},
                     "by_length": {b: summarize([r for r in rows_a if bucket(r["n_removed"], cap) == b])
                                   for b in ("le64", "65_128", f"129_{cap}")},
                     "cross_cos_flat": c_flat, "cross_cos_per_position": c_pos, "per_example": rows_a}}
        if step % 100 == 0:
            rows_b, gists_b = eval_set(hold_b, der_b)
            cb_flat, cb_pos = pairwise_cos(gists_b)
            res["B"] = {"overall": summarize(rows_b), "cross_cos_flat": cb_flat, "cross_cos_per_position": cb_pos,
                        "by_length": {b: summarize([r for r in rows_b if bucket(r["n_removed"], cap) == b])
                                      for b in ("le64", "65_128", f"129_{cap}")}, "per_example": rows_b}
        if step % 500 == 0:
            from transformers import AutoModelForCausalLM
            from common import MODEL_PATH
            eager = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32,
                                                         attn_implementation="eager").to(device).eval()
            exs = [hold_a[i] for i in attn_idx]
            fa, f2 = attention_fraction(eager, exs, [gists_a[i] for i in attn_idx], device)
            res["attention"] = {"n_examples": len(exs), "gist_pos_1_16": fa, "gist_pos_2_16": f2}
            del eager
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ent_a = res["A"]["overall"]["entropy"]
        res["entropy_warning"] = ent_a < ENTROPY_WARN
        if res["entropy_warning"]:
            print(f"WARNING: held-out mean rollout entropy {ent_a:.4f} < {ENTROPY_WARN} nats at step {step} "
                  f"(possible collapse to hard tokens)")
        for v in (res["A"]["overall"]["kl_gist_gap"], res["A"]["overall"]["kl_gist_nogap"]):
            if not math.isfinite(v):
                sys.exit(f"STOP: non-finite held-out KL at step {step}")
        res["eval_seconds"] = time.time() - t_ev
        eval_rows.append(res)
        with open(eval_log, "a") as f:
            f.write(json.dumps(res) + "\n")

        rec = res["A"]["overall"]["recovery_mean_ratio"]
        improved = rec > best["recovery"]
        if improved:
            best.update(recovery=rec, step=step)
        state = {"step": step, "micro_consumed": micro_consumed, "trainable": trainable_state(encoder, embedding2),
                 "optimizer": optimizer.state_dict(), "best": dict(best), "config": CONFIG}
        atomic_save(state, latest)
        if improved:
            link_or_replace(latest, os.path.join(ckpt_dir, "best.pt"))
        if step > 0 and step % PERMANENT_EVERY == 0:
            link_or_replace(latest, os.path.join(ckpt_dir, f"step_{step:05d}.pt"))
        save_plots(train_rows, eval_rows, nogist_mean_a, out_dir, sched)
        o = res["A"]["overall"]
        msg = (f"EVAL step {step}: A KL_gist gap {o['kl_gist_gap']:.4f} (nogist {o['kl_nogist_gap']:.4f}) "
               f"nogap {o['kl_gist_nogap']:.4f} | recovery {rec:.4f} (ratio-of-means {o['recovery_ratio_of_means']:.4f}) "
               f"| per-source " + " ".join(f"{s}={res['A']['by_source'][s].get('recovery_mean_ratio', float('nan')):.3f}"
                                            for s in ("sni", "rlvr", "squad")) +
               f" | swap-own {o['swap_minus_own_gap']:+.4f} | cos {c_flat:.3f} | ent {o['entropy']:.3f} "
               f"| top1 {o['distinct_top1']:.1f}")
        if "B" in res:
            ob = res["B"]["overall"]
            msg += f" || B recovery {ob['recovery_mean_ratio']:.4f} swap-own {ob['swap_minus_own_gap']:+.4f}"
        if "attention" in res:
            msg += f" || attn 1-16 {res['attention']['gist_pos_1_16']:.4f} 2-16 {res['attention']['gist_pos_2_16']:.4f}"
        print(msg + f"  [{res['eval_seconds']:.0f}s]{'  (new best)' if improved else ''}")

    if start_step == 0:
        check_sync(0)
        on_main(evaluate, 0)

    print(f"GPUs={world} L={L} micro={MICRO_BATCH} accum={ACCUM} (per GPU) peak_lr={PEAK_LR} min_lr={MIN_LR} warmup={WARMUP_STEPS} "
          f"schedule={sched} steps={args.steps} betas={BETAS} eps={EPS} wd={WEIGHT_DECAY} clip={MAX_NORM} "
          f"entropy_coef={args.entropy_coef} ckpt_dir={ckpt_dir}")
    step_times = []
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        lr = lr_at(step, sched)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        epochs, step_exs = set(), []
        for _k in range(GLOBAL_BATCH):
            idx, ep = order(micro_consumed * MICRO_BATCH + _k)
            step_exs.append(train[idx])
            epochs.add(ep)
        micro_consumed += GLOBAL_BATCH // MICRO_BATCH  # counts micro-batches over all GPUs
        srcs = [e["source"] for e in step_exs]
        n_tok = sum(len(e["response_ids"]) for e in step_exs)  # loss = token-weighted mean over the whole step
        n_micro = GLOBAL_BATCH // MICRO_BATCH
        mine = step_exs[rank * MICRO_BATCH * ACCUM:(rank + 1) * MICRO_BATCH * ACCUM]
        stats = torch.zeros(3, dtype=torch.float64, device=device)  # [sum of kd shares, sum of entropies, non-finite]
        for m in range(ACCUM):
            exs = mine[m * MICRO_BATCH:(m + 1) * MICRO_BATCH]
            loss, kd, ent, _ = micro_loss(models, exs, capture, device, args.entropy_coef, n_tok, n_micro)
            if not (math.isfinite(loss.item()) and math.isfinite(kd.item())):
                stats[2] += 1
                break
            loss.backward()
            stats[0] += kd.item()
            stats[1] += ent.item()
        if dist_on:
            dist.all_reduce(stats)
        if stats[2] > 0:
            sys.exit(f"STOP: non-finite training loss at step {step}")
        if dist_on:
            for p in params:
                dist.all_reduce(p.grad)  # sum: each rank's loss is already its share of the step mean
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=MAX_NORM).item()
        if not math.isfinite(gn):
            sys.exit(f"STOP: non-finite gradient norm at step {step}: {gn}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)
        row = {"step": step, "train_kd": stats[0].item(), "train_entropy": stats[1].item() / n_micro,
               "grad_norm_preclip": gn, "clip_active": gn > MAX_NORM, "lr": lr, "epoch": sorted(epochs),
               "n_sni": srcs.count("sni"), "n_rlvr": srcs.count("rlvr"), "n_squad": srcs.count("squad"),
               "time": time.time(), "step_seconds": dt}
        train_rows.append(row)
        if is_main:
            with open(train_log, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"step {step:5d}  kd={row['train_kd']:.5f}  ent={row['train_entropy']:.3f}  grad_norm={gn:.3e}  "
              f"clip={'on' if row['clip_active'] else 'off'}  lr={lr:.2e}  {dt:.1f}s")
        if is_eval_step(step) and not args.skip_evals:
            check_sync(step)
            on_main(evaluate, step)

    if step_times:
        sps = sum(step_times) / len(step_times)
        n_evals = sum(is_eval_step(s) for s in range(0, TOTAL_STEPS + 1))
        ev = [r["eval_seconds"] for r in eval_rows]
        print(f"TIMING: {sps:.2f} s per optimizer step (mean over {len(step_times)} steps); "
              f"{TOTAL_STEPS} steps = {sps * TOTAL_STEPS / 3600:.1f} h of training + {n_evals} evals "
              f"(step-0 eval took {ev[0]:.0f}s)" if ev else "")
    if dist_on:
        check_sync(args.steps)
        print("GPU weight sync OK at the end")
        dist.destroy_process_group()
    print(f"done. best held-out A recovery {best['recovery']:.4f} @ step {best['step']}")


CONFIG = {"num_trainable_layers": NUM_LAYERS, "L": L, "peak_lr": PEAK_LR, "min_lr": MIN_LR,
          "warmup_steps": WARMUP_STEPS, "betas": BETAS, "eps": EPS, "weight_decay": WEIGHT_DECAY,
          "micro_batch": MICRO_BATCH, "accum": ACCUM, "world_size": 1, "max_norm": MAX_NORM, "total_steps": TOTAL_STEPS, "seed": SEED}

if __name__ == "__main__":
    main()
