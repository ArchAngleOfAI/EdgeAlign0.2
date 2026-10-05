"""Failure-mode analysis + example episodes for the Stage 1 report.

Reads /data/.../rollouts/stage1/*.jsonl, writes icae_rl/stage1/analysis.json and
icae_rl/stage1/examples.md (a few transcripts per condition).
"""

import json
import re
from collections import Counter
from pathlib import Path

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response
from bfcl_eval.model_handler.utils import default_decode_execute_prompting

ROLL = Path("/data/a84460786/edgealign_icae_rl/rollouts/stage1")
OUT = Path(__file__).resolve().parent / "stage1"


def call_kind(text):
    """How BFCL's prompting decoder sees one generated response."""
    try:
        d = default_decode_execute_prompting(text, has_tool_call_tag=False)
        if is_empty_execute_response(d):
            return "decoded_empty"
        if any(re.search(r"\w\(\)", c) for c in d) and re.search(r"\w\((?!\s*\w+\s*=)['\"\d\[]", text):
            return "decoded_positional_args_dropped"
        return "decoded_ok"
    except Exception:
        pass
    t = text.strip()
    if '"name"' in t and ('"parameters"' in t or '"arguments"' in t):
        return "fail_json_style_call"
    if re.match(r"^\[?\w+\(.*\)\]?", t) and "\n" in t:
        return "fail_call_plus_extra_text"
    if "\\_" in t:
        return "fail_escaped_underscore"
    if not re.search(r"\w\(", t):
        return "fail_prose_no_call"
    return "fail_other"


def main():
    OUT.mkdir(exist_ok=True)
    analysis, ex = {}, ["# Stage 1 example episodes (greedy)\n",
                        "Truncated transcripts. `valid` = BFCL success; `turns` = per-turn pass.\n"]
    for f in sorted(p for p in ROLL.glob("*.jsonl") if not p.name.endswith(".partial.jsonl")):
        eps = [json.loads(l) for l in f.open()]
        kinds = Counter(call_kind(c["text"]) for e in eps for c in e["calls"])
        n = sum(kinds.values())
        analysis[f.stem] = {"n_calls": n, "kinds": dict(kinds),
                            "frac": {k: v / n for k, v in kinds.items()},
                            "errors": dict(Counter(str(e["error_type"]) for e in eps)),
                            "any_turn_pass": sum(any(e["turn_pass"]) for e in eps)}
        # examples: first successful episode (if any) + the first failed one
        picks = [e for e in eps if e["valid"]][:1] + [e for e in eps if not e["valid"]][:1]
        for e in picks:
            ex.append(f"\n## {f.stem}: {e['id']} (valid={e['valid']}, turns={e['turn_pass']}, "
                      f"error={e['error_type']})\n")
            ci = 0
            for mm in e["messages"]:
                if mm["kind"] == "query":
                    ex.append(f"- **USER (turn {mm['turn']})**: {mm['content'][:300]}")
                elif mm["kind"] == "call":
                    c = e["calls"][ci]
                    ci += 1
                    ex.append(f"  - MODEL [hist {c['hist_tokens']} tok -> {c['slots']} slots, "
                              f"{call_kind(c['text'])}]: `{c['text'][:250]!r}`")
                else:
                    ex.append(f"  - TOOL: `{mm['content'][:200]!r}`")
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))
    (OUT / "examples.md").write_text("\n".join(ex) + "\n")
    for k, v in analysis.items():
        print(k, v["n_calls"], {a: round(b, 3) for a, b in sorted(v["frac"].items())},
              "any_turn_pass", v["any_turn_pass"])


if __name__ == "__main__":
    main()
