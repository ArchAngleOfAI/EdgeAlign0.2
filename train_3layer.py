"""Long soft-prompt distillation run: last 3 encoder layers trainable, L=16, warmup + clipping.

Settings (see RUN_NOTES_3layer.md): Qwen3-0.6B student encoder / frozen teacher /
frozen receiver; trainable = Embedding2 + untied LM_head + the encoder's LAST 3
decoder layers; soft-prompt length 16; prompt length 128; fp32; seed 0; AdamW,
peak LR 1e-4, linear warmup 0 -> 1e-4 over the first 100 optimizer steps then
constant; global batch 16 = micro-batch 2 x grad-accum 8 (each micro loss / 8);
clip_grad_norm_(max_norm=1.0) once per optimizer step, after accumulation;
10,000 optimizer steps; full backprop through the rollout (no detaching).

Reuse, not reimplementation: model loading/unfreezing, the trainable-param list,
the forward pass (generate_softprompt -> receiver_forward -> KD loss) and the
batch stream all come from diagnostics/_common.py, which itself imports
build_softprompt_generator, generate_softprompt, receiver_forward and
fineweb_edu_batches from the original modules and holds the verbatim KD loss.

Held-out set (never seen in training): 8 batches x 4 prompts x 128 tokens,
packed by the SAME fineweb_edu_batches code but pointed at a DIFFERENT parquet
shard (sample/10BT/001_00000) than the one training reads (000_00000, which
fineweb_edu_batches hard-codes). At startup `check_no_overlap` also confirms
that no held-out source document (by id or by text hash) appears in the
training shard's row groups that this run can reach.

Outputs: logs/train_3layer.jsonl (one line per optimizer step),
logs/eval_3layer.jsonl (one line per eval), the loss-curve PNG at the repo root
(rewritten atomically at every eval), checkpoints under --ckpt-dir
(latest.pt every eval, best.pt by held-out loss, step_XXXXX.pt every 1000
steps). --resume continues from latest.pt, including the LR-schedule position
and the data-stream position. --smoke redirects every output to a scratch
location so a smoke run never touches the real run's files.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "diagnostics"))

import smoke_train  # noqa: E402
from _common import (  # noqa: E402
    PROMPT_LEN, batch_stream, forward_loss, get_device, get_tokenizer, load_all, setup_stdout,
    trainable_params,
)

NUM_LAYERS = 3
L = 16
PEAK_LR = 1e-4
WARMUP_STEPS = 100
MICRO_BATCH = 2
ACCUM = 8
MAX_NORM = 1.0
TOTAL_STEPS = 10_000
EVAL_BATCHES, EVAL_BATCH_SIZE = 8, 4
PERMANENT_EVERY = 1000
SEED = 0

HELDOUT_SHARD_URL = smoke_train.FINEWEB_EDU_SHARD_URL.replace("000_00000.parquet", "001_00000.parquet")
PNG_NAME = "kd_loss_curve_3layer_lr1e-4_bs16_accum8_gradclip.png"
DEFAULT_CKPT_DIR = "/data/a84460786/testfolder_checkpoints/train_3layer"

COLOR_TRAIN = "#2a78d6"    # categorical slot 1 (training loss: raw + moving average)
COLOR_EVAL = "#eb6834"     # categorical slot 2 (held-out loss)
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def lr_at(step):
    """LR used for optimizer step `step` (1-indexed): linear 0 -> PEAK over WARMUP_STEPS, then constant."""
    return PEAK_LR * min(step, WARMUP_STEPS) / WARMUP_STEPS


def is_eval_step(step):
    return step == 0 or (step <= 300 and step % 10 == 0) or step % 100 == 0


# ---------------------------------------------------------------- data

def heldout_batches(tokenizer):
    """8 x (4, 128) batches from shard 001_00000, packed by the unmodified fineweb_edu_batches.

    fineweb_edu_batches reads smoke_train.FINEWEB_EDU_SHARD_URL when the generator
    first runs, so the URL is swapped only while these 8 batches are drawn, then restored.
    """
    original = smoke_train.FINEWEB_EDU_SHARD_URL
    smoke_train.FINEWEB_EDU_SHARD_URL = HELDOUT_SHARD_URL
    try:
        gen = smoke_train.fineweb_edu_batches(tokenizer, PROMPT_LEN, batch_size=EVAL_BATCH_SIZE)
        batches = [next(gen) for _ in range(EVAL_BATCHES)]
        gen.close()
    finally:
        smoke_train.FINEWEB_EDU_SHARD_URL = original
    assert smoke_train.FINEWEB_EDU_SHARD_URL == original
    return batches


def check_no_overlap(tokenizer, total_steps):
    """Held-out source docs vs. every training-shard row group this run can reach: no shared id or text."""
    import fsspec
    import pyarrow.parquet as pq

    held = pq.ParquetFile(fsspec.open(HELDOUT_SHARD_URL, "rb").open()).read_row_group(0, columns=["id", "text"])
    # Docs actually used: enough leading docs to cover the held-out tokens (plus one for the EOS-split tail).
    need, used, n_docs = EVAL_BATCHES * EVAL_BATCH_SIZE * PROMPT_LEN, 0, 0
    for text in held.column("text").to_pylist():
        used += len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 1
        n_docs += 1
        if used >= need:
            break
    held_ids = set(held.column("id").to_pylist()[:n_docs])
    held_hash = {hashlib.sha1(t.encode()).hexdigest() for t in held.column("text").to_pylist()[:n_docs]}

    train_pf = pq.ParquetFile(fsspec.open(smoke_train.FINEWEB_EDU_SHARD_URL, "rb").open())
    train_tokens_needed = total_steps * ACCUM * MICRO_BATCH * PROMPT_LEN
    covered, rg, docs_checked = 0, 0, 0
    # token_count (GPT-2 tokens) approximates Qwen tokens; check 2x the need for margin.
    while covered < 2 * train_tokens_needed and rg < train_pf.num_row_groups:
        t = train_pf.read_row_group(rg, columns=["id", "text", "token_count"])
        ids = t.column("id").to_pylist()
        hashes = {hashlib.sha1(x.encode()).hexdigest() for x in t.column("text").to_pylist()}
        assert not held_ids & set(ids), f"held-out doc id found in training shard row group {rg}"
        assert not held_hash & hashes, f"held-out doc text found in training shard row group {rg}"
        covered += sum(t.column("token_count").to_pylist())
        docs_checked += len(ids)
        rg += 1
    msg = (f"no-overlap check passed: {n_docs} held-out source docs (shard 001_00000, row group 0) vs "
           f"{docs_checked:,} training docs in row groups 0-{rg - 1} of shard 000_00000 "
           f"(~{covered / 1e6:.1f}M GPT-2 tokens, >= 2x the {train_tokens_needed / 1e6:.1f}M tokens "
           f"{total_steps} steps consume); no shared id or text hash")
    print(msg)
    return msg


# ---------------------------------------------------------------- checkpoints

def trainable_state(encoder, embedding2):
    return {
        "embedding2": embedding2.state_dict(),
        "lm_head": encoder.lm_head.state_dict(),
        "last_layers": [layer.state_dict() for layer in encoder.model.layers[-NUM_LAYERS:]],
    }


def load_trainable_state(encoder, embedding2, state):
    embedding2.load_state_dict(state["embedding2"])
    encoder.lm_head.load_state_dict(state["lm_head"])
    for layer, sd in zip(encoder.model.layers[-NUM_LAYERS:], state["last_layers"]):
        layer.load_state_dict(sd)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def link_or_replace(src, dst):
    """Point dst at src's current file (hard link: no extra write, survives src being replaced later)."""
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    os.link(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------- logging / plot

def truncate_jsonl(path, max_step):
    """On resume: drop lines logged after the checkpoint we resumed from."""
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r["step"] <= max_step]
    with open(path + ".tmp", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(path + ".tmp", path)
    return rows


def moving_average(values, window=50):
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def save_plot(train_rows, eval_rows, path, total_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    if train_rows:
        s = [r["step"] for r in train_rows]
        v = [r["train_loss"] for r in train_rows]
        ax.plot(s, v, color=COLOR_TRAIN, lw=0.8, alpha=0.3, label="train loss (per step, 16 prompts)")
        ax.plot(s, moving_average(v), color=COLOR_TRAIN, lw=2,
                label="train loss, trailing moving average (50 steps; fewer before step 50)")
    if eval_rows:
        ax.plot([r["step"] for r in eval_rows], [r["heldout_mean"] for r in eval_rows], color=COLOR_EVAL,
                lw=2, marker="o", ms=5, markeredgecolor=COLOR_SURFACE, markeredgewidth=1,
                label="held-out loss (mean of 32 prompts, L=16)")
    ax.axvline(WARMUP_STEPS, color=COLOR_MUTED, ls="--", lw=1.2, label=f"end of LR warmup (step {WARMUP_STEPS})")
    ax.set_xlim(0, max(total_steps, 1))
    ax.set_xlabel("optimizer step", color="#3d3c38")
    ax.set_ylabel("KD loss  KL(teacher || receiver), T=1", color="#3d3c38")
    ax.set_title("Soft-prompt distillation — last 3 layers + LM_head + Embedding2 trainable, L=16\n"
                 "AdamW peak LR 1e-4 (100-step linear warmup, then constant), global batch 16 "
                 "(2 × accum 8), grad clip 1.0", fontsize=11)
    ax.grid(True, color=COLOR_GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=COLOR_SURFACE)
    plt.close(fig)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main

def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--resume", action="store_true", help="Continue from <ckpt-dir>/latest.pt.")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="Write logs/plot/checkpoints to scratch locations (logs/smoke_3layer/, "
                         "<ckpt-dir>_smoke) instead of the real run's files.")
    args = ap.parse_args()

    if args.smoke:
        log_dir = os.path.join(REPO_ROOT, "logs", "smoke_3layer")
        png_path = os.path.join(log_dir, PNG_NAME)
        ckpt_dir = args.ckpt_dir.rstrip("/") + "_smoke"
    else:
        log_dir = os.path.join(REPO_ROOT, "logs")
        png_path = os.path.join(REPO_ROOT, PNG_NAME)
        ckpt_dir = args.ckpt_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_log = os.path.join(log_dir, "train_3layer.jsonl")
    eval_log = os.path.join(log_dir, "eval_3layer.jsonl")
    latest = os.path.join(ckpt_dir, "latest.pt")

    torch.manual_seed(SEED)
    device = get_device()
    encoder, embedding2, _, teacher, receiver = load_all(device, num_trainable_layers=NUM_LAYERS)
    encoder.config.tie_word_embeddings = False  # latent re-tie risk from report.md test 3
    ptrs = {encoder.lm_head.weight.data_ptr(), encoder.model.embed_tokens.weight.data_ptr(),
            embedding2.weight.data_ptr()}
    assert len(ptrs) == 3, "lm_head / embed_tokens / embedding2 must be three distinct tensors"
    params = trainable_params(encoder, embedding2, NUM_LAYERS)
    trainable_ids = {id(p) for p in params}
    assert all(p.requires_grad == (id(p) in trainable_ids) for p in encoder.parameters()), \
        "encoder requires_grad set does not match the trainable list"
    assert not any(p.requires_grad for p in teacher.parameters()) and \
        not any(p.requires_grad for p in receiver.parameters())
    print(f"trainable: embedding2 + lm_head + last {NUM_LAYERS} layers = "
          f"{sum(p.numel() for p in params):,} params; distinct data_ptrs OK; "
          f"tie_word_embeddings={encoder.config.tie_word_embeddings}")
    optimizer = torch.optim.AdamW(params, lr=lr_at(1))

    tok = get_tokenizer()
    overlap_msg = check_no_overlap(tok, args.steps)
    eval_batches = heldout_batches(tok)

    start_step, micro_consumed = 0, 0
    best = {"heldout_mean": math.inf, "step": None}
    train_rows, eval_rows = [], []
    if args.resume:
        ckpt = torch.load(latest, map_location=device)
        load_trainable_state(encoder, embedding2, ckpt["trainable"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, micro_consumed, best = ckpt["step"], ckpt["micro_consumed"], ckpt["best"]
        train_rows = truncate_jsonl(train_log, start_step)
        eval_rows = truncate_jsonl(eval_log, start_step)
        print(f"resumed from {latest}: step {start_step}, {micro_consumed} micro-batches consumed, "
              f"best held-out {best['heldout_mean']:.6f} @ step {best['step']}")
    else:
        for p in (train_log, eval_log):
            if os.path.exists(p):
                os.remove(p)

    stream = batch_stream(tok, MICRO_BATCH)
    t_ff = time.time()
    for _ in range(micro_consumed):  # restore the data-stream position
        next(stream)
    if micro_consumed:
        print(f"fast-forwarded data stream by {micro_consumed} micro-batches in {time.time() - t_ff:.0f}s")

    @torch.no_grad()
    def evaluate(step):
        vals = [forward_loss(encoder, embedding2, teacher, receiver, b.to(device), L).item()
                for b in eval_batches]
        mean = sum(vals) / len(vals)
        if not all(math.isfinite(v) for v in vals):
            sys.exit(f"STOP: non-finite held-out loss at step {step}: {vals}")
        row = {"step": step, "heldout_mean": mean, "heldout_per_batch": vals, "time": time.time()}
        eval_rows.append(row)
        with open(eval_log, "a") as f:
            f.write(json.dumps(row) + "\n")

        state = {"step": step, "micro_consumed": micro_consumed, "trainable": trainable_state(encoder, embedding2),
                 "optimizer": optimizer.state_dict(), "best": best, "config": CONFIG}
        improved = mean < best["heldout_mean"]
        if improved:
            best.update(heldout_mean=mean, step=step)
            state["best"] = best
        atomic_save(state, latest)
        if improved:
            link_or_replace(latest, os.path.join(ckpt_dir, "best.pt"))
        if step > 0 and step % PERMANENT_EVERY == 0:
            link_or_replace(latest, os.path.join(ckpt_dir, f"step_{step:05d}.pt"))
        save_plot(train_rows, eval_rows, png_path, args.steps)
        print(f"EVAL step {step}: held-out mean={mean:.6f} per-batch={[round(v, 4) for v in vals]}"
              f"{'  (new best)' if improved else ''}")

    if start_step == 0:
        evaluate(0)

    print(f"L={L} micro_batch={MICRO_BATCH} accum={ACCUM} global_batch={MICRO_BATCH * ACCUM} "
          f"peak_lr={PEAK_LR} warmup={WARMUP_STEPS} clip={MAX_NORM} steps={args.steps} ckpt_dir={ckpt_dir}")
    step_times = []
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(ACCUM):
            ids = next(stream).to(device)
            micro_consumed += 1
            loss = forward_loss(encoder, embedding2, teacher, receiver, ids, L)
            if not math.isfinite(loss.item()):
                sys.exit(f"STOP: non-finite training loss at step {step} (micro-batch {len(micro_losses) + 1})")
            (loss / ACCUM).backward()
            micro_losses.append(loss.item())
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=MAX_NORM).item()
        if not math.isfinite(gn):
            sys.exit(f"STOP: non-finite gradient norm at step {step}: {gn}")
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)

        row = {"step": step, "train_loss": sum(micro_losses) / ACCUM, "grad_norm_preclip": gn,
               "clip_active": gn > MAX_NORM, "lr": lr, "time": time.time(), "step_seconds": dt}
        train_rows.append(row)
        with open(train_log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {step:5d}  train_loss={row['train_loss']:.6f}  grad_norm={gn:.3e}  "
              f"clip={'on' if row['clip_active'] else 'off'}  lr={lr:.2e}  {dt:.1f}s")

        if is_eval_step(step):
            evaluate(step)

    if step_times:
        n_evals = sum(is_eval_step(s) for s in range(0, TOTAL_STEPS + 1))
        sps = sum(step_times) / len(step_times)
        print(f"TIMING: {sps:.2f} s per optimizer step (mean over {len(step_times)} steps); "
              f"estimated {TOTAL_STEPS:,} steps = {sps * TOTAL_STEPS / 3600:.1f} h of training "
              f"(+ {n_evals} evals/checkpoints)")
    print(f"done. best held-out {best['heldout_mean']:.6f} @ step {best['step']}. {overlap_msg}")


CONFIG = {"num_trainable_layers": NUM_LAYERS, "L": L, "peak_lr": PEAK_LR, "warmup_steps": WARMUP_STEPS,
          "micro_batch": MICRO_BATCH, "accum": ACCUM, "max_norm": MAX_NORM, "prompt_len": PROMPT_LEN,
          "seed": SEED, "heldout_shard": HELDOUT_SHARD_URL, "train_shard": smoke_train.FINEWEB_EDU_SHARD_URL}

if __name__ == "__main__":
    main()
