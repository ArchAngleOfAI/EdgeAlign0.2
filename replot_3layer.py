"""Re-plot the train_3layer loss curve over only the steps trained so far.

Read-only with respect to the live run: it reads logs/train_3layer.jsonl and
logs/eval_3layer.jsonl and writes a SEPARATE PNG (the run keeps rewriting its
own full-range PNG). Reuses train_3layer.save_plot, with the x-axis ending at
the last logged step instead of 10,000. CPU-only; safe to run while training.

Usage: python replot_3layer.py [--out PATH]
"""

import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU the run is using

from train_3layer import REPO_ROOT, save_plot  # noqa: E402

DEFAULT_OUT = os.path.join(REPO_ROOT, "kd_loss_curve_3layer_lr1e-4_bs16_accum8_gradclip_trained_range.png")


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line mid-write by the live run
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    train = read_jsonl(os.path.join(REPO_ROOT, "logs", "train_3layer.jsonl"))
    evals = read_jsonl(os.path.join(REPO_ROOT, "logs", "eval_3layer.jsonl"))
    last = max([r["step"] for r in train] + [r["step"] for r in evals] + [1])
    save_plot(train, evals, args.out, last)
    print(f"wrote {args.out} (steps 0-{last}, {len(train)} train rows, {len(evals)} evals)")


if __name__ == "__main__":
    main()
