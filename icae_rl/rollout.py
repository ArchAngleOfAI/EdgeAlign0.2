"""Decoder inputs per condition + batched BFCL episode runner.

Chat format = Mistral-7B-Instruct-v0.2 template, built piecewise so that every condition sees
the SAME token ids for its uncompressed parts:
  P0 (system)        = <s> ▁[INST] tok(SYSTEM + "\n\n")
  first user message = tok(content) ▁[/INST]            (continues the [INST] opened in P0)
  other user message = ▁[INST] tok(content) ▁[/INST]    (user queries and tool results)
  assistant message  = tok(content) </s>
(the official template merges the system prompt into the first user message; tok() adds the
sentencepiece prefix space, which matches the template's spaces.)

At a model call in turn t, step k: H_prev = messages before the current user query,
q = the current user query, H_cur = this turn's earlier calls and tool results.
  full      : P0 + H_prev + q + H_cur                       (H_cur ends with ▁[/INST])
  icae      : P0 + [slots(H_prev)][FT] + q + [slots(H_cur minus final ▁[/INST])][FT] + ▁[/INST]
              (spec's literal reading: everything except system prompt + current user message
              is compressed; the final ▁[/INST] is kept as the generation cue)
  icae_turn : P0 + [slots(H_prev)][FT] + q + H_cur          (other reading: only earlier TURNS
              are compressed; this turn's calls/tool outputs stay uncompressed)
  nohist    : P0 + q                                        (nothing inserted)
q is rendered as a first user message whenever H_prev is not shown as text.
Pretrained ICAE (adapter "pre"): no FT token after the slots (its LM objective reads
[slots] -> continuation), and the encoder input starts with BOS as in its pretraining;
fine-tuned ICAE: no BOS in the encoder input, as in instruct_ft_tokenize_function.
"""

import time
import uuid

import torch

import bfcl_env
from icae_model import BOS, EOS, FT_ROW, INST_CLOSE, INST_OPEN, n_slots

CONDITIONS = ("full", "icae", "icae_turn", "nohist")


class Renderer:
    def __init__(self, model):
        self.m = model
        self._cache = {}

    def tok(self, text):
        hit = self._cache.get(text)
        if hit is None:
            hit = self.m.ids(text)
            if len(self._cache) < 200000:
                self._cache[text] = hit
        return hit

    def msgs(self, msgs, opened):
        """opened: the first message continues an already-open [INST] (only if it is a user msg)."""
        out = []
        for i, mm in enumerate(msgs):
            if mm["role"] == "user":
                if i == 0 and opened:
                    out += self.tok(mm["content"]) + INST_CLOSE
                else:
                    out += INST_OPEN + self.tok(mm["content"]) + INST_CLOSE
            else:
                out += self.tok(mm["content"]) + [EOS]
        return out

    def p0(self, sys_text):
        return [BOS] + INST_OPEN + self.tok(sys_text + "\n\n")


def call_spec(r, sys_text, state, cond):
    """Token-level description of one decoder input (no tensors), saved with the episode so the
    RL teacher-forced pass can rebuild it exactly.
    Returns {"p0": ids, "parts": [("ids", ids) | ("hist", ids) | ("ft",)], ...}."""
    msgs = state["messages"]
    qi = max(i for i, mm in enumerate(msgs) if mm["kind"] == "query")
    h_prev, q, h_cur = msgs[:qi], msgs[qi], msgs[qi + 1:]
    p0 = r.p0(sys_text)
    parts = []
    hist_tokens = 0
    if cond == "full":
        parts.append(("ids", r.msgs(h_prev, opened=True) + r.msgs([q], opened=(qi == 0))
                      + r.msgs(h_cur, opened=False)))
    elif cond == "nohist":
        parts.append(("ids", r.msgs([q], opened=True)))
    elif cond in ("icae", "icae_turn"):
        if h_prev:
            hp = r.msgs(h_prev, opened=True)
            parts += [("hist", hp), ("ft",)]
            hist_tokens += len(hp)
        parts.append(("ids", r.msgs([q], opened=True)))
        if h_cur:
            hc = r.msgs(h_cur, opened=False)
            if cond == "icae_turn":
                parts.append(("ids", hc))
            else:
                assert hc[-len(INST_CLOSE):] == INST_CLOSE
                hc = hc[:-len(INST_CLOSE)]
                parts += [("hist", hc), ("ft",), ("ids", INST_CLOSE)]
                hist_tokens += len(hc)
    else:
        raise ValueError(cond)
    return {"p0": p0, "parts": parts, "hist_tokens": hist_tokens,
            "turn": state["turn"], "step": state["step"]}


def build_inputs(model, specs, adapter, grad=False, slot_override=None):
    """specs -> list of decoder input embeddings. Compresses all histories of all specs in one
    batched encoder pass. adapter: "ft" / "pre" / RL adapter name (ignored for text-only specs).
    slot_override: optional list (per spec) of lists of precomputed slot tensors, used for the
    swap test. Returns (inputs, slots_per_spec)."""
    use_ft = adapter != "pre"
    hists, where = [], []
    for si, sp in enumerate(specs):
        for pi, part in enumerate(sp["parts"]):
            if part[0] == "hist":
                ids = part[1] if use_ft else [BOS] + part[1]
                hists.append(ids)
                where.append((si, pi))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        slots = model.compress(hists, adapter) if hists else []
        by = {w: s for w, s in zip(where, slots)}
        inputs, slots_per_spec = [], []
        for si, sp in enumerate(specs):
            parts, mine, k = [("ids", sp["p0"])], [], 0
            for pi, part in enumerate(sp["parts"]):
                if part[0] == "ids":
                    parts.append(part)
                elif part[0] == "hist":
                    s = by[(si, pi)]
                    if slot_override is not None and slot_override[si] is not None:
                        s = slot_override[si][min(k, len(slot_override[si]) - 1)]
                    k += 1
                    mine.append(s)
                    parts.append(("emb", s))
                elif part[0] == "ft":
                    if use_ft:
                        parts.append(("emb", model.special(adapter, FT_ROW)[None]))
            inputs.append(model.assemble(parts))
            slots_per_spec.append(mine)
    return inputs, slots_per_spec


def run_episodes(model, tasks, task_ids, cond, adapter="ft", batch_size=8, greedy=True,
                 temperature=0.7, top_p=0.95, max_new_tokens=512, seed=0, log=print,
                 slot_source=None, keep_specs=False, repeats=1, parser="strict",
                 on_done=None):
    """Runs one episode per (task id x repeats) under `cond`; returns list of episode records.
    slot_source: optional {task_id: other_task_id} for the swap test (slots for task i are
    computed from task j's recorded histories, see swap_histories)."""
    r = Renderer(model)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    queue = [(tid, rep) for tid in task_ids for rep in range(repeats)]
    active, done = [], []
    t_start = time.time()
    while queue or active:
        while queue and len(active) < batch_size:
            tid, rep = queue.pop(0)
            entry, gt = tasks[tid]
            uid = f"ep{uuid.uuid4().hex[:12]}"
            g = bfcl_env.episode(entry, uid, parser)
            ep = {"id": tid, "rep": rep, "uid": uid, "gen": g, "state": next(g),
                  "sys": bfcl_env.system_prompt(entry), "calls": [], "t0": time.time()}
            active.append(ep)
        specs = [call_spec(r, ep["sys"], ep["state"], cond) for ep in active]
        override = None
        if slot_source is not None:
            override = [slot_source(ep, sp) for ep, sp in zip(active, specs)]
        inputs, _ = build_inputs(model, specs, adapter, slot_override=override)
        kw = dict(greedy=greedy, temperature=temperature, top_p=top_p,
                  max_new_tokens=max_new_tokens, generator=gen)
        try:
            outs = model.generate(inputs, **kw)
        except torch.OutOfMemoryError:
            # one long conversation inflates the padded batch cache: retry one prompt at a time
            torch.cuda.empty_cache()
            if log:
                log(f"OOM at batch {len(inputs)} (max len {max(x.shape[0] for x in inputs)}); "
                    f"retrying one at a time")
            outs = [model.generate([x], **kw)[0] for x in inputs]
        still = []
        for ep, sp, x, o in zip(active, specs, inputs, outs):
            text = model.tok.decode(o, skip_special_tokens=True)
            rec = {"turn": sp["turn"], "step": sp["step"], "dec_len": int(x.shape[0]),
                   "hist_tokens": sp["hist_tokens"],
                   "slots": sum(n_slots(len(p[1])) for p in sp["parts"] if p[0] == "hist"),
                   "gen_ids": o, "text": text}
            if keep_specs:
                rec["spec"] = sp
            ep["calls"].append(rec)
            try:
                ep["state"] = ep["gen"].send(text)
                still.append(ep)
            except StopIteration as stop:
                res = stop.value
                entry, gt = tasks[ep["id"]]
                ev = bfcl_env.evaluate(entry, gt, res["model_response"], ep["uid"], parser)
                bfcl_env.cleanup_instances(ep["uid"])
                done.append({"id": ep["id"], "rep": ep["rep"], "cond": cond, "adapter": adapter,
                             "parser": parser,
                             "valid": ev["valid"], "error_type": ev.get("error_type"),
                             "turn_pass": ev["turn_pass"],
                             "turn_pass_frac": ev["turn_pass_frac"],
                             "force_quit": res["force_quit"], "n_calls": res["n_calls"],
                             "seconds": time.time() - ep["t0"],
                             "messages": res["messages"], "calls": ep["calls"]})
                if on_done:
                    on_done(done[-1])
                if log:
                    n_ok = sum(d["valid"] for d in done)
                    log(f"[{cond}/{adapter}] {len(done)}/{len(task_ids) * repeats} "
                        f"{ep['id']} valid={ev['valid']} turns={ev['turn_pass']} "
                        f"calls={res['n_calls']} ({n_ok} ok, "
                        f"{(time.time() - t_start) / len(done):.1f} s/ep)")
        active = still
    return done
