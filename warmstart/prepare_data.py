"""Step 1: offline data preparation for the warm-start gist pretraining.

Stages (run in order; each reads the previous stage's files in DATA_DIR):
  build     -> candidates.jsonl   : user messages from SNI / RLVR-IFeval / SQuAD, length filters,
                                    SNI span CAP, RLVR constraint location + decontamination
  generate  -> responses_<k>.jsonl: responses from the frozen instruct Qwen3-0.6B (HF generate,
                                    vLLM is not installed), one shard per GPU
  klscore   -> kl_<k>.jsonl       : KL_nogist (teacher vs receiver with the removed part deleted,
                                    nothing inserted) on the response tokens, fp32
  finalize  -> train.jsonl, heldout_a.jsonl, heldout_b.jsonl in DATA_DIR, plus
               warmstart/data_stats.json and warmstart/data_sample.jsonl in the repo

Layout (removed part always LAST in the user message):
  A  SNI:   "Input:\n{input}\n\nTask:\n" + definition
  B  RLVR:  (message without the constraint) + "\n\n" + constraint
  C  SQuAD: "Context:\n{passage}\n\nQuestion: " + question
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DATA_DIR, ENDOFTEXT, IM_END, MAX_NEW_TOKENS, MAX_SEQ, MIN_KEPT, MIN_REMOVED, RAW_DIR, encode,
    get_tokenizer, pieces,
)

REPO_WS = os.path.dirname(os.path.abspath(__file__))
SNI_DIR = os.path.join(RAW_DIR, "natural-instructions")
SEED = 0
SNI_MAX_PER_TASK = 150
CAP_HARD = 256
NOGAP_FRAC = 0.15
KL_DROP_FRAC = 0.30
MIX = {"sni": 0.5, "rlvr": 0.3, "squad": 0.2}
HELDOUT_A = 256
HELDOUT_B = 128
# How many candidates to generate responses for (the user chose: keep 50/30/20, total set by
# RLVR's size -- all usable RLVR prompts are used, SNI and SQuAD are sized to match).
GEN_SNI, GEN_SQUAD = 26000, 11000
HOLD_POOL = {"sni": 320, "rlvr": 200, "squad": 140}
HOLD_B_PER_TASK = 3
LANG_TYPES = ("All Lowercase", "All Uppercase")  # both say "Your entire response should be in English"

CAND = os.path.join(DATA_DIR, "candidates.jsonl")
STATS_BUILD = os.path.join(DATA_DIR, "stats_build.json")


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def pct(values, qs=(50, 75, 90, 95, 99)):
    if not len(values):
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def dist(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            **{f"p{q}": float(np.percentile(v, q)) for q in (10, 25, 30, 50, 75, 90)}, "max": float(v.max())}


# ---------------------------------------------------------------- build

def make_example(ex_id, source, task, user_kept, removed, split, counters):
    """Tokenize the pieces separately; apply the removed/kept/prompt-length rules."""
    prefix_ids, removed_ids, suffix_ids = pieces(user_kept, removed)
    n_kept = len(encode(user_kept))
    if not (MIN_REMOVED <= len(removed_ids) <= counters["_cap"]):
        counters["drop_removed_len"] += 1
        return None
    if n_kept < MIN_KEPT:
        counters["drop_kept_len"] += 1
        return None
    if len(prefix_ids) + len(removed_ids) + len(suffix_ids) + 1 > MAX_SEQ:
        counters["drop_total_len_prompt"] += 1
        return None
    return {"id": ex_id, "source": source, "task": task, "split": split, "user_kept": user_kept,
            "removed_part": removed, "user_full": user_kept + removed,
            "prefix_ids": prefix_ids, "removed_ids": removed_ids, "suffix_ids": suffix_ids,
            "n_kept_tokens": n_kept}


def load_sni_task(name):
    with open(os.path.join(SNI_DIR, "tasks", name + ".json")) as f:
        return json.load(f)


def sni_english(d):
    return d["Input_language"] == ["English"] and d["Output_language"] == ["English"]


def build_sni(stats, rng):
    split_dir = os.path.join(SNI_DIR, "splits", "default")
    train_tasks = [l.strip() for l in open(os.path.join(split_dir, "train_tasks.txt")) if l.strip()]
    test_tasks = [l.strip() for l in open(os.path.join(split_dir, "test_tasks.txt")) if l.strip()]
    assert not set(train_tasks) & set(test_tasks)

    tasks = {}
    non_en = {"train": 0, "test": 0}
    multi_def = []
    for split, names in (("train", train_tasks), ("test", test_tasks)):
        for n in names:
            d = load_sni_task(n)
            if not sni_english(d):
                non_en[split] += 1
                continue
            if len(d["Definition"]) != 1:  # task288 has 2 alternatives; SNI's own code uses [0]
                multi_def.append(n)
            tasks[n] = (split, d)
    train_en = [n for n, (s, _) in tasks.items() if s == "train"]
    test_en = [n for n, (s, _) in tasks.items() if s == "test"]
    def_len = {n: len(encode(tasks[n][1]["Definition"][0])) for n in tasks}
    lens = [def_len[n] for n in train_en]
    p95 = float(np.percentile(lens, 95))
    cap = min(CAP_HARD, int(math.ceil(p95 / 16) * 16))
    over = [n for n in train_en if def_len[n] > cap]
    under = [n for n in train_en if def_len[n] < MIN_REMOVED]
    test_over = [n for n in test_en if def_len[n] > cap]
    test_under = [n for n in test_en if def_len[n] < MIN_REMOVED]
    stats["sni"] = {
        "repo": "github.com/allenai/natural-instructions", "files": "tasks/*.json, splits/default/{train,test}_tasks.txt",
        "fields_used": "Definition[0], Input_language, Output_language, Instances[].{id,input}",
        "train_tasks_listed": len(train_tasks), "test_tasks_listed": len(test_tasks),
        "non_english_dropped": non_en, "tasks_with_multiple_definitions_used_first": multi_def, "train_tasks_english": len(train_en), "test_tasks_english": len(test_en),
        "definition_tokens_train_pool": {"n_tasks": len(lens), **pct(lens), "min": min(lens), "max": max(lens),
                                         "mean": float(np.mean(lens))},
        "CAP": cap, "CAP_rule": f"p95={p95:.1f} rounded up to a multiple of 16, capped at {CAP_HARD}",
        "train_tasks_dropped_over_cap": len(over), "train_tasks_dropped_under_8": len(under),
        "test_tasks_dropped_over_cap": len(test_over), "test_tasks_dropped_under_8": len(test_under),
    }
    print(json.dumps(stats["sni"], indent=1))
    return tasks, def_len, cap, set(over) | set(under), set(test_over) | set(test_under)


def sni_examples(tasks, dropped, dropped_test, counters):
    train_rows, hold_pool, test_rows = [], [], []
    for name, (split, d) in sorted(tasks.items()):
        if (split == "train" and name in dropped) or (split == "test" and name in dropped_test):
            continue
        definition = d["Definition"][0]
        inst = list(d["Instances"])
        random.Random(f"{SEED}-{name}").shuffle(inst)
        if split == "train":
            groups = (("train", inst[:SNI_MAX_PER_TASK], train_rows), ("heldout_a", inst[SNI_MAX_PER_TASK:SNI_MAX_PER_TASK + 3], hold_pool))
        else:
            groups = (("heldout_b", inst[:HOLD_B_PER_TASK], test_rows),)
        for sp, items, out in groups:
            for it in items:
                kept = f"Input:\n{it['input']}\n\nTask:\n"
                e = make_example(f"sni-{it['id']}", "sni", name, kept, definition, sp, counters[sp])
                if e:
                    out.append(e)
    return train_rows, hold_pool, test_rows


# RLVR-IFeval: the `constraint` field is a TEMPLATE ({N}, {end phrase}, "at least / around / at
# most" ...). The literal text is recovered by filling the template from `ground_truth` and requiring
# the match to be anchored at the very start or very end of the message (the two places the
# dataset puts constraints). Anything else is "not located reliably" and dropped.
_GT_KEYS = {"N": "N", "i": "i", "word": "word", "letter": "letter", "end phrase": "end_phrase",
            "first word": "first_word", "postscript marker": "postscript_marker",
            "section splitter": "section_splitter", "options": "options"}


def _ws(p):
    s = r"\s+".join(re.escape(w) for w in p.split())
    if p and p[0].isspace():
        s = r"\s+" + s if s else r"\s+"
    if p and p[-1].isspace() and p.strip():
        s += r"\s+"
    return s


def constraint_regex(template, gt):
    out = []
    for p in re.split(r"(\{[^}]*\}|at least / around / at most)", template):
        if not p:
            continue
        if p == "at least / around / at most":
            out.append(r"(?:at least|around|at most)")
        elif p.startswith("{"):
            k = p[1:-1]
            if k in _GT_KEYS and gt.get(_GT_KEYS[k]) not in (None, ""):
                out.append(_ws(str(gt[_GT_KEYS[k]])))
            elif k == "forbidden words" and gt.get("forbidden_words"):
                out.append(r",\s*".join(re.escape(w) for w in gt["forbidden_words"]))
            elif k in ("keyword1", "keyword2") and gt.get("keyword_list"):
                out.append(re.escape(gt["keyword_list"][0 if k == "keyword1" else 1]))
            else:
                return None
        else:
            out.append(_ws(p))
    return "".join(out)


def locate_constraint(msg, template, gt):
    """Return (rest_of_message, literal_constraint) or None."""
    for tm in (template, template[5:] if template.startswith("From ") else None):
        if tm is None:
            continue
        rx = constraint_regex(tm, gt)
        if rx is None:
            continue
        a = re.match(r"\s*(" + rx + r")", msg, flags=re.S)
        b = re.search(r"(" + rx + r")\s*$", msg, flags=re.S)
        if a and not b:
            return msg[a.end(1):].strip(), a.group(1).strip()
        if b and not a:
            return msg[:b.start(1)].strip(), b.group(1).strip()
    return None


def ngrams13(text):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + 13]) for i in range(len(w) - 12)}


def build_rlvr(stats, rng, counters):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(RAW_DIR, "rlvr_ifeval.parquet")).to_pylist()
    ifeval = pq.read_table(os.path.join(RAW_DIR, "ifeval.parquet")).column("prompt").to_pylist()
    ifbench = pq.read_table(os.path.join(RAW_DIR, "ifbench_test.parquet")).column("prompt").to_pylist()
    grams_ifeval = set().union(*(ngrams13(p) for p in ifeval))
    grams_ifbench = set().union(*(ngrams13(p) for p in ifbench))
    s = {"n_total": len(rows), "types": dict(collections.Counter(r["constraint_type"] for r in rows)),
         "excluded_language_types": list(LANG_TYPES),
         "excluded_language_reason": "no constraint_type is about response language as such; these two are "
                                     "the only types whose constraint text sets the response language "
                                     "('Your entire response should be in English, ...')",
         "excluded_language_n": 0, "not_located": collections.Counter(), "decontam_ifeval": 0,
         "decontam_ifbench": 0, "decontam_both": 0, "duplicate_user_full": 0,
         "decontam_sources": {"google/IFEval": len(ifeval), "allenai/IFBench_test": len(ifbench)},
         "decontam_rule": "13-gram (lowercased \\w+ words) overlap between the task text (message with the "
                          "constraint removed) and any IFEval / IFBench_test prompt",
         "whole_message_would_hit": 0}
    out, seen = [], set()
    for k, r in enumerate(rows):
        if r["constraint_type"] in LANG_TYPES:
            s["excluded_language_n"] += 1
            continue
        assert len(r["messages"]) == 1 and r["messages"][0]["role"] == "user"
        msg = r["messages"][0]["content"]
        loc = locate_constraint(msg, r["constraint"], json.loads(r["ground_truth"]))
        if loc is None:
            s["not_located"][r["constraint_type"]] += 1
            continue
        rest, constraint = loc
        # User's choice (2026-09-30): 13-gram check on the TASK TEXT only (message minus the
        # constraint). RLVR copies IFEval's constraint templates verbatim, so a whole-message check
        # drops prompts for template wording alone; the whole-message count is logged for reference.
        g_full = ngrams13(msg)
        s["whole_message_would_hit"] += bool(g_full & (grams_ifeval | grams_ifbench))
        g = ngrams13(rest)
        hit_e, hit_b = bool(g & grams_ifeval), bool(g & grams_ifbench)
        if hit_e or hit_b:
            s["decontam_ifeval"] += hit_e
            s["decontam_ifbench"] += hit_b
            s["decontam_both"] += hit_e and hit_b
            continue
        kept = rest + "\n\n"
        if kept + constraint in seen:
            s["duplicate_user_full"] += 1
            continue
        seen.add(kept + constraint)
        out.append((f"rlvr-{k}", r["constraint_type"], kept, constraint))
    s["not_located"] = dict(s["not_located"])
    s["not_located_total"] = sum(s["not_located"].values())
    s["decontam_dropped_total"] = s["decontam_ifeval"] + s["decontam_ifbench"] - s["decontam_both"]
    rng.shuffle(out)
    hold_ids = {x[0] for x in out[:HOLD_POOL["rlvr"]]}
    train_rows, hold_rows = [], []
    for ex_id, ctype, kept, constraint in out:
        sp = "heldout_a" if ex_id in hold_ids else "train"
        e = make_example(ex_id, "rlvr", ctype, kept, constraint, sp, counters[sp])
        if e:
            (hold_rows if sp == "heldout_a" else train_rows).append(e)
    s["usable_after_location_decontam_dedup"] = len(out)
    stats["rlvr"] = s
    print(json.dumps(s, indent=1, default=str))
    return train_rows, hold_rows


def build_squad(stats, rng, counters):
    import pyarrow.parquet as pq
    train = pq.read_table(os.path.join(RAW_DIR, "squad_train.parquet")).to_pylist()
    val = pq.read_table(os.path.join(RAW_DIR, "squad_val.parquet")).to_pylist()
    rng.shuffle(train)
    rng.shuffle(val)
    out = {"train": [], "heldout_a": []}
    for sp, rows, n in (("train", train, GEN_SQUAD), ("heldout_a", val, HOLD_POOL["squad"])):
        for r in rows:
            if len(out[sp]) >= n:
                break
            kept = f"Context:\n{r['context']}\n\nQuestion: "
            e = make_example(f"squad-{r['id']}", "squad", None, kept, r["question"].strip(), sp, counters[sp])
            if e:
                out[sp].append(e)
    stats["squad"] = {"train_split_rows": len(train), "validation_split_rows": len(val),
                      "train_candidates": len(out["train"]), "heldout_candidates": len(out["heldout_a"])}
    return out["train"], out["heldout_a"]


def cmd_build(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = random.Random(SEED)
    stats = {}
    tasks, def_len, cap, dropped, dropped_test = build_sni(stats, rng)
    counters = {sp: collections.Counter(_cap=cap) for sp in ("train", "heldout_a", "heldout_b")}
    per_source_counters = {}

    snap = lambda: {sp: dict(c) for sp, c in counters.items()}  # noqa: E731
    sni_train, sni_hold, sni_test = sni_examples(tasks, dropped, dropped_test, counters)
    per_source_counters["sni"] = snap()
    rng_sni = random.Random(SEED + 1)
    rng_sni.shuffle(sni_train)
    stats["sni"]["train_pool_after_length_filters"] = len(sni_train)
    sni_train = sni_train[:GEN_SNI]
    counters = {sp: collections.Counter(_cap=cap) for sp in counters}
    rlvr_train, rlvr_hold = build_rlvr(stats, rng, counters)
    per_source_counters["rlvr"] = snap()
    counters = {sp: collections.Counter(_cap=cap) for sp in counters}
    squad_train, squad_hold = build_squad(stats, rng, counters)
    per_source_counters["squad"] = snap()
    stats["length_filter_drops_at_build"] = per_source_counters

    rng.shuffle(sni_hold)
    rows = sni_train + sni_hold[:HOLD_POOL["sni"]] + sni_test + rlvr_train + rlvr_hold + squad_train + squad_hold
    write_jsonl(CAND, rows)
    stats["candidates"] = {f"{s}/{sp}": n for (s, sp), n in
                           collections.Counter((r["source"], r["split"]) for r in rows).items()}
    json.dump(stats, open(STATS_BUILD, "w"), indent=1, default=str)
    print(json.dumps(stats["candidates"], indent=1))


# ---------------------------------------------------------------- generate

def cmd_generate(args):
    import torch
    from transformers import AutoModelForCausalLM
    from common import MODEL_PATH

    rows = read_jsonl(CAND)
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    # --nsub splits a shard's remaining work further (used when a shard's GPU was too slow)
    rows = [r for i, r in enumerate(rows) if i % args.nsub == args.sub]
    suffix = f"{args.shard}" if args.nsub == 1 else f"{args.shard}_{args.sub}"
    out_path = os.path.join(DATA_DIR, f"responses_{suffix}.jsonl")
    done = set()
    for p in glob.glob(os.path.join(DATA_DIR, "responses_*.jsonl")):
        done |= {r["id"] for r in read_jsonl(p)}
    rows = [r for r in rows if r["id"] not in done]
    rows.sort(key=lambda r: -(len(r["prefix_ids"]) + len(r["removed_ids"]) + len(r["suffix_ids"])))
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} to generate ({len(done)} already done)", flush=True)

    torch.manual_seed(SEED + args.shard)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda().eval()
    f = open(out_path, "a")
    i = 0
    import time
    t0 = time.time()
    while i < len(rows):
        plen = len(rows[i]["prefix_ids"]) + len(rows[i]["removed_ids"]) + len(rows[i]["suffix_ids"])
        bs = max(8, min(args.batch, int(args.batch * 512 / max(plen, 1))))
        batch = rows[i:i + bs]
        seqs = [r["prefix_ids"] + r["removed_ids"] + r["suffix_ids"] for r in batch]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), ENDOFTEXT, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, s in enumerate(seqs):  # left padding for generation
            ids[k, T - len(s):] = torch.tensor(s)
            mask[k, T - len(s):] = 1
        with torch.no_grad():
            gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), do_sample=True,
                                 temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=MAX_NEW_TOKENS,
                                 eos_token_id=[IM_END, ENDOFTEXT], pad_token_id=ENDOFTEXT)
        new = gen[:, T:].tolist()
        for r, toks in zip(batch, new):
            finish = "length"
            for j, t in enumerate(toks):
                if t in (IM_END, ENDOFTEXT):
                    finish = "im_end" if t == IM_END else "endoftext"
                    toks = toks[:j + 1] if t == IM_END else toks[:j]
                    break
            f.write(json.dumps({"id": r["id"], "response_ids": toks, "finish": finish}) + "\n")
        f.flush()
        i += len(batch)
        el = time.time() - t0
        print(f"shard {args.shard}: {i}/{len(rows)}  {el / 60:.1f} min  eta {el / i * (len(rows) - i) / 60:.1f} min",
              flush=True)
    f.close()


# ---------------------------------------------------------------- degenerate-response rule

def degenerate(resp):
    """Returns the drop reason or None. response_ids end with <|im_end|> when finish == im_end."""
    if resp["finish"] != "im_end":
        return "no_end_of_turn"
    content = resp["response_ids"][:-1]
    text = get_tokenizer().decode(content)
    if not text.strip():
        return "empty"
    if "<think>" in text or "</think>" in text:
        return "think_tag"
    if len(content) >= 16:
        grams = [tuple(content[i:i + 4]) for i in range(len(content) - 3)]
        if len(set(grams)) / len(grams) < 0.5:
            return "repetition"
    return None


def load_joined():
    cands = {r["id"]: r for r in read_jsonl(CAND)}
    resp = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "responses_*.jsonl"))):
        for r in read_jsonl(p):
            resp[r["id"]] = r
    return cands, resp


# ---------------------------------------------------------------- KL_nogist

def cmd_klscore(args):
    import torch
    from transformers import AutoModelForCausalLM
    from common import MODEL_PATH, build_batch, kd_per_token, per_example_kl, receiver_response_logits, \
        teacher_response_logits

    cands, resp = load_joined()
    todo = []
    for k, (cid, r) in enumerate(sorted(cands.items())):
        if k % args.nshards != args.shard or cid not in resp or degenerate(resp[cid]):
            continue
        e = dict(r, response_ids=resp[cid]["response_ids"], gap=True)
        if len(e["prefix_ids"]) + len(e["removed_ids"]) + len(e["suffix_ids"]) + len(e["response_ids"]) > MAX_SEQ:
            continue
        todo.append(e)
    out_path = os.path.join(DATA_DIR, f"kl_{args.shard}.jsonl")
    done = {r["id"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()
    todo = [e for e in todo if e["id"] not in done]
    todo.sort(key=lambda e: -(len(e["prefix_ids"]) + len(e["removed_ids"]) + len(e["suffix_ids"]) + len(e["response_ids"])))
    print(f"klscore shard {args.shard}: {len(todo)} to score", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).cuda().eval()
    dev = torch.device("cuda")
    f = open(out_path, "a")
    import time
    t0, i = time.time(), 0
    while i < len(todo):
        n = len(todo[i]["prefix_ids"]) + len(todo[i]["removed_ids"]) + len(todo[i]["suffix_ids"]) + len(todo[i]["response_ids"])
        bs = max(1, min(args.batch, int(args.batch * 512 / n)))
        batch_ex = todo[i:i + bs]
        with torch.no_grad():
            b = build_batch(batch_ex, dev)
            t_log = teacher_response_logits(model, b)
            empty = [torch.zeros((0, model.config.hidden_size), device=dev) for _ in batch_ex]
            r_log = receiver_response_logits(model, b, empty)
            kl = per_example_kl(kd_per_token(r_log, t_log), b["resp_len"])
        for e, v in zip(batch_ex, kl):
            f.write(json.dumps({"id": e["id"], "kl_nogist": v}) + "\n")
        f.flush()
        i += len(batch_ex)
        if (i // len(batch_ex)) % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"klscore shard {args.shard}: {i}/{len(todo)} {el / 60:.1f} min eta {el / i * (len(todo) - i) / 60:.1f} min",
                  flush=True)
    f.close()


# ---------------------------------------------------------------- finalize

def cmd_finalize(args):
    stats = json.load(open(STATS_BUILD))
    tok = get_tokenizer()
    cands, resp = load_joined()
    kl = {}
    for p in glob.glob(os.path.join(DATA_DIR, "kl_*.jsonl")):
        for r in read_jsonl(p):
            kl[r["id"]] = r["kl_nogist"]

    gen_counts = collections.Counter()
    pools = collections.defaultdict(list)
    for cid, c in cands.items():
        key = (c["source"], c["split"])
        gen_counts[f"{key[0]}/{key[1]}/candidates"] += 1
        if cid not in resp:
            gen_counts[f"{key[0]}/{key[1]}/missing_response"] += 1
            continue
        reason = degenerate(resp[cid])
        if reason:
            gen_counts[f"{key[0]}/{key[1]}/drop_{reason}"] += 1
            continue
        e = dict(c, response_ids=resp[cid]["response_ids"])
        total = len(e["prefix_ids"]) + len(e["removed_ids"]) + len(e["suffix_ids"]) + len(e["response_ids"])
        if total > MAX_SEQ:
            gen_counts[f"{key[0]}/{key[1]}/drop_total_len_gt_{MAX_SEQ}"] += 1
            continue
        assert cid in kl, f"no KL_nogist for {cid}"
        e["kl_nogist"] = kl[cid]
        e["response"] = tok.decode(e["response_ids"][:-1])
        e["n_removed_tokens"] = len(e["removed_ids"])
        e["n_response_tokens"] = len(e["response_ids"])
        e["n_teacher_tokens"] = total
        pools[key].append(e)
    stats["response_generation"] = {
        "backend": "HF transformers generate (vLLM not installed), bf16, left padding",
        "sampling": "temperature 0.7, top_p 0.8, top_k 20, max_new_tokens 256, eos = <|im_end|> or <|endoftext|>",
        "input": "teacher token ids (prefix + removed + suffix), i.e. the FULL user message, non-thinking template",
        "degenerate_rule": "drop if (a) no <|im_end|> within 256 new tokens (incl. stopping on <|endoftext|>), "
                           "(b) empty/whitespace-only content, (c) '<think>' or '</think>' in content, "
                           "(d) >= 16 content tokens and distinct-4-gram ratio < 0.5 (heavy repetition)",
        "counts": dict(sorted(gen_counts.items())),
    }

    rng = random.Random(SEED)
    kl_stats = {}
    final = {}
    thresholds = {}
    for src in ("sni", "rlvr", "squad"):
        pool = sorted(pools[(src, "train")], key=lambda e: e["id"])
        rng.shuffle(pool)
        P = len(pool)
        # no-gap share chosen so that no-gap = 15% of the FINAL set (gap examples lose 30% to the filter)
        n_ng = int(round(P * NOGAP_FRAC / (NOGAP_FRAC + (1 - NOGAP_FRAC) / (1 - KL_DROP_FRAC))))
        nogap, gap = pool[:n_ng], pool[n_ng:]
        kls = [e["kl_nogist"] for e in gap]
        thr = float(np.percentile(kls, KL_DROP_FRAC * 100))
        thresholds[src] = thr
        gap_kept = [e for e in gap if e["kl_nogist"] > thr]
        for e in nogap:
            e["gap"] = False
        for e in gap_kept:
            e["gap"] = True
        kl_stats[src] = {"gap_before_filter": dist(kls), "gap_after_filter": dist([e["kl_nogist"] for e in gap_kept]),
                         "threshold_p30": thr, "gap_dropped": len(gap) - len(gap_kept), "nogap_kl_nogist_unfiltered":
                         dist([e["kl_nogist"] for e in nogap])}
        final[src] = (nogap, gap_kept)

    # Mix 50/30/20 (user's choice): RLVR is the limiting source, SNI and SQuAD are subsampled to match.
    n_b = len(final["rlvr"][0]) + len(final["rlvr"][1])
    target = {"rlvr": n_b, "sni": int(round(n_b * MIX["sni"] / MIX["rlvr"])),
              "squad": int(round(n_b * MIX["squad"] / MIX["rlvr"]))}
    train = []
    for src in ("sni", "rlvr", "squad"):
        nogap, gap = final[src]
        have = len(nogap) + len(gap)
        assert have >= target[src], f"{src}: only {have} after filtering, need {target[src]}"
        n_ng = int(round(target[src] * NOGAP_FRAC))
        train += nogap[:n_ng] + gap[:target[src] - n_ng]
    rng.shuffle(train)

    # Held-out A: 256, ~50/30/20, ~15% no-gap, same filters (KL threshold = the training pool's
    # per-source p30). Held-out B: 128 unseen-task SNI gap examples, spread over tasks.
    hold_a = []
    quota = {"sni": 128, "rlvr": 77, "squad": 51}
    for src, q in quota.items():
        pool = sorted(pools[(src, "heldout_a")], key=lambda e: e["id"])
        rng.shuffle(pool)
        n_ng = int(round(q * NOGAP_FRAC))
        ng = pool[:n_ng]
        gp = [e for e in pool[n_ng:] if e["kl_nogist"] > thresholds[src]][:q - n_ng]
        assert len(gp) == q - n_ng, f"held-out A {src}: only {len(gp)} gap examples pass"
        for e in ng:
            e["gap"] = False
        for e in gp:
            e["gap"] = True
        hold_a += ng + gp
    pool_b = sorted(pools[("sni", "heldout_b")], key=lambda e: e["id"])
    rng.shuffle(pool_b)
    pool_b = [e for e in pool_b if e["kl_nogist"] > thresholds["sni"]]
    by_task = collections.defaultdict(list)
    for e in pool_b:
        by_task[e["task"]].append(e)
    hold_b, rnd = [], 0
    while len(hold_b) < HELDOUT_B:
        added = False
        for t in sorted(by_task):
            if rnd < len(by_task[t]) and len(hold_b) < HELDOUT_B:
                hold_b.append(by_task[t][rnd])
                added = True
        assert added, f"held-out B: only {len(hold_b)} examples available"
        rnd += 1
    for e in hold_b:
        e["gap"] = True

    # Disjointness checks
    ids_train = {e["id"] for e in train}
    assert not ids_train & {e["id"] for e in hold_a}
    assert not ids_train & {e["id"] for e in hold_b}
    assert not {e["task"] for e in hold_b} & {e["task"] for e in train if e["source"] == "sni"}
    assert not {e["user_full"] for e in train} & {e["user_full"] for e in hold_a + hold_b}

    keep = ["id", "source", "task", "user_full", "user_kept", "removed_part", "gap", "response",
            "n_kept_tokens", "n_removed_tokens", "n_response_tokens", "n_teacher_tokens", "kl_nogist",
            "prefix_ids", "removed_ids", "suffix_ids", "response_ids"]
    for name, rows in (("train", train), ("heldout_a", hold_a), ("heldout_b", hold_b)):
        write_jsonl(os.path.join(DATA_DIR, f"{name}.jsonl"), [{k: e[k] for k in keep} for e in rows])

    def summary(rows):
        c = collections.Counter((e["source"], e["gap"]) for e in rows)
        by = {s: {"total": c[(s, True)] + c[(s, False)], "gap": c[(s, True)], "nogap": c[(s, False)]}
              for s in ("sni", "rlvr", "squad")}
        n = len(rows)
        return {"total": n, "per_source": by,
                "fractions": {s: round(by[s]["total"] / n, 4) if n else 0 for s in by},
                "nogap_fraction": round(sum(1 for e in rows if not e["gap"]) / n, 4) if n else 0,
                "removed_tokens": dist([e["n_removed_tokens"] for e in rows]),
                "response_tokens": dist([e["n_response_tokens"] for e in rows]),
                "teacher_tokens": dist([e["n_teacher_tokens"] for e in rows])}

    stats["kl_nogist_filter"] = {"rule": "per source, gap examples with KL_nogist <= the 30th percentile of that "
                                         "source's gap pool are dropped; no-gap examples are not filtered; "
                                         "held-out sets use the same per-source thresholds (held-out B uses SNI's)",
                                 "per_source": kl_stats}
    stats["final"] = {"train": summary(train), "heldout_a": summary(hold_a), "heldout_b": summary(hold_b),
                      "heldout_b_tasks": len({e["task"] for e in hold_b}),
                      "train_sni_tasks": len({e["task"] for e in train if e["source"] == "sni"}),
                      "train_sni_max_instances_per_task": max(collections.Counter(
                          e["task"] for e in train if e["source"] == "sni").values()),
                      "mix_rule": "user's choice (2026-09-30): keep 50/30/20; RLVR limits the size, all of its "
                                  "filtered examples are used and SNI/SQuAD are subsampled to 5/3 and 2/3 of it",
                      "nogap_rule": "no-gap = 15% of the final set per source (marked before the KL filter at the "
                                    "share that leaves 15% after it)"}
    for name, rows in (("heldout_a", hold_a), ("heldout_b", hold_b)):
        stats["final"][name]["kl_nogist_gap"] = dist([e["kl_nogist"] for e in rows if e["gap"]])
    for d in (REPO_WS, DATA_DIR):
        json.dump(stats, open(os.path.join(d, "data_stats.json"), "w"), indent=1, default=str)

    sample = []
    for src in ("sni", "rlvr", "squad"):
        rows = [e for e in train if e["source"] == src]
        sample += [{k: e[k] for k in keep if not k.endswith("_ids")} for e in rows[:10]]
    write_jsonl(os.path.join(REPO_WS, "data_sample.jsonl"), sample)
    print(json.dumps(stats["final"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "generate", "klscore", "finalize"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--nsub", type=int, default=1)
    args = ap.parse_args()
    {"build": cmd_build, "generate": cmd_generate, "klscore": cmd_klscore, "finalize": cmd_finalize}[args.stage](args)


if __name__ == "__main__":
    main()
