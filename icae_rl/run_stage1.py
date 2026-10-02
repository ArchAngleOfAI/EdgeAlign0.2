"""Stage 1 feasibility: run every condition on TRAIN and TEST (greedy), no training.

Runs (cond, adapter): full, icae/ft, icae/pre, nohist, icae_turn/ft. One jsonl per
(run, split) under /data/.../rollouts/stage1/ (full transcripts); finished files are skipped,
so the script can simply be restarted. `--summarize` writes icae_rl/stage1/summary.json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

ROLL = Path("/data/a84460786/edgealign_icae_rl/rollouts/stage1")
HERE = Path(__file__).resolve().parent
RUNS = [("full", "ft"), ("icae", "ft"), ("icae", "pre"), ("nohist", "ft"), ("icae_turn", "ft")]


def run_name(cond, adapter):
    if cond in ("full", "nohist"):
        return cond
    return f"{cond}_{adapter}"


def summarize():
    import bfcl_env
    split = bfcl_env.load_split()
    out = {"note": "BFCL multi_turn_base, split by task id 60/40 seed 0. RL trains on TRAIN, so "
                   "these scores are NOT comparable to the public leaderboard.", "runs": {}}
    for cond, adapter in RUNS:
        name = run_name(cond, adapter)
        for sp in ("train", "test"):
            f = ROLL / f"{name}_{sp}.jsonl"
            if not f.exists():
                continue
            eps = [json.loads(l) for l in f.open()]
            if len(eps) != len(split[sp]):
                continue
            calls = [c for e in eps for c in e["calls"]]
            hist_calls = [c for c in calls if c["hist_tokens"] > 0]
            meta = json.loads((ROLL / f"{name}_{sp}.meta.json").read_text())
            out["runs"][f"{name}/{sp}"] = {
                "n": len(eps),
                "success": sum(e["valid"] for e in eps),
                "success_rate": sum(e["valid"] for e in eps) / len(eps),
                "mean_turn_pass_frac": statistics.mean(e["turn_pass_frac"] for e in eps),
                "force_quit": sum(e["force_quit"] for e in eps),
                "mean_calls_per_ep": statistics.mean(e["n_calls"] for e in eps),
                "mean_hist_tokens_per_call": statistics.mean(c["hist_tokens"] for c in calls),
                "mean_slots_per_call": statistics.mean(c["slots"] for c in calls),
                "mean_hist_tokens_per_call_with_hist": (statistics.mean(
                    c["hist_tokens"] for c in hist_calls) if hist_calls else 0),
                "mean_slots_per_call_with_hist": (statistics.mean(
                    c["slots"] for c in hist_calls) if hist_calls else 0),
                "max_hist_tokens": max(c["hist_tokens"] for c in calls),
                "mean_decoder_len": statistics.mean(c["dec_len"] for c in calls),
                "mean_gen_tokens": statistics.mean(len(c["gen_ids"]) for c in calls),
                "mean_wall_s_per_ep_concurrent": statistics.mean(e["seconds"] for e in eps),
                "throughput_s_per_ep": meta["wall_s"] / max(1, len(eps) - meta.get("n_resumed", 0)),
                "batch_size": meta["batch_size"],
                "error_types": {k: sum(e.get("error_type") == k for e in eps)
                                for k in sorted({str(e.get("error_type")) for e in eps})},
            }
    # FULL-context history length (what ICAE has to compress), from the full run
    (HERE / "stage1").mkdir(exist_ok=True)
    (HERE / "stage1" / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--runs", default="all")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--parser", default="strict", choices=["strict", "lenient"],
                    help="lenient = diagnostic, NOT BFCL's parser (see bfcl_env.lenient_decode)")
    args = ap.parse_args()
    if args.summarize:
        return summarize()
    import torch
    import bfcl_env
    import rollout
    from icae_model import ICAE
    ROLL.mkdir(parents=True, exist_ok=True)
    tasks = bfcl_env.load_tasks()
    split = bfcl_env.load_split()
    runs = RUNS if args.runs == "all" else [tuple(r.split("/")) for r in args.runs.split(",")]
    m = ICAE("cuda", {"ft": "ft", "pre": "pre"})
    for sp in args.splits.split(","):
        for cond, adapter in runs:
            name = run_name(cond, adapter) + ("_lenient" if args.parser == "lenient" else "")
            f = ROLL / f"{name}_{sp}.jsonl"
            ids = split[sp][:args.limit] if args.limit else split[sp]
            if f.exists() and sum(1 for _ in f.open()) == len(ids):
                print(f"skip {f} (done)", flush=True)
                continue
            # finished episodes are appended to a .partial file as they complete; a restart
            # resumes from it (wall time then covers only the resumed part, noted in meta)
            part = ROLL / f"{name}_{sp}.partial.jsonl"
            prev = [json.loads(l) for l in part.open()] if part.exists() else []
            todo = [t for t in ids if t not in {e["id"] for e in prev}]
            print(f"=== {name} on {sp} ({len(ids)} tasks, {len(prev)} resumed)", flush=True)
            t0 = time.time()
            with part.open("a") as ph:
                def on_done(e):
                    ph.write(json.dumps(e) + "\n")
                    ph.flush()
                eps = prev + rollout.run_episodes(
                    m, tasks, todo, cond, adapter, batch_size=args.batch_size, greedy=True,
                    log=lambda s: print(s, flush=True), parser=args.parser, on_done=on_done)
            wall = time.time() - t0
            order = {t: i for i, t in enumerate(ids)}
            eps.sort(key=lambda e: order[e["id"]])
            with f.open("w") as fh:
                for e in eps:
                    fh.write(json.dumps(e) + "\n")
            (ROLL / f"{name}_{sp}.meta.json").write_text(json.dumps(
                {"wall_s": wall, "batch_size": args.batch_size, "n": len(eps), "n_resumed": len(prev),
                 "gpu": torch.cuda.get_device_name(), "max_new_tokens": 512,
                 "parser": args.parser}))
            print(f"=== {name} on {sp}: {sum(e['valid'] for e in eps)}/{len(eps)} valid, "
                  f"{wall:.0f} s", flush=True)
    summarize()


if __name__ == "__main__":
    main()
