"""BFCL multi_turn_base environment for icae_rl.

Mirrors BFCL's own prompting pipeline (`BaseHandler.inference_multi_turn_prompting` +
`_evaluate_single_multi_turn_entry`), with the model call left open so that the caller can
build the decoder input itself (full text, ICAE slots, no history, ...).

Message format (BFCL prompting mode, as BFCL's prompting handlers do for models without a tool
role, e.g. `api_inference/mistral.py`):
  - system prompt = `system_prompt_pre_processing_chat_model` (default format
    "ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic"),
  - assistant message = the raw generated text,
  - tool results = ONE user message, `format_execution_results_prompting(...)` (repr of a list of
    {"role": "tool", "name": call, "content": result}).
"""

import copy
import json
import random
import re
from pathlib import Path

from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    response_checker,
    state_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import (
    default_decode_execute_prompting,
    format_execution_results_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

CATEGORY = "multi_turn_base"
HERE = Path(__file__).resolve().parent
SPLIT_PATH = HERE / "split.json"


def lenient_decode(text):
    """NOT BFCL's parser (diagnostic only): un-escape markdown underscores, then return the first
    bracketed span that BFCL's own decoder parses into a non-empty call list (ignoring any prose
    around it). Falls back to BFCL's decoder on the whole text (which may raise)."""
    t = text.replace("\\_", "_")
    for i, ch in enumerate(t):
        if ch != "[":
            continue
        depth = 0
        for j in range(i, len(t)):
            depth += {"[": 1, "]": -1}.get(t[j], 0)
            if depth == 0:
                try:
                    d = default_decode_execute_prompting(t[i:j + 1], has_tool_call_tag=False)
                    if not is_empty_execute_response(d):
                        return d
                except Exception:
                    pass
                break
    return default_decode_execute_prompting(t, has_tool_call_tag=False)


def decode(text, parser="strict"):
    if parser == "lenient":
        return lenient_decode(text)
    return default_decode_execute_prompting(text, has_tool_call_tag=False)


def load_tasks():
    """Returns {task_id: (entry, ground_truth)} for multi_turn_base, entries with function docs."""
    entries = load_dataset_entry(CATEGORY)
    gts = {g["id"]: g["ground_truth"] for g in load_ground_truth_entry(CATEGORY)}
    return {e["id"]: (e, gts[e["id"]]) for e in entries}


def make_split(task_ids, seed=0, train_frac=0.6):
    ids = sorted(task_ids, key=lambda s: int(s.rsplit("_", 1)[1]))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = round(train_frac * len(ids))
    train = sorted(ids[:n_train], key=lambda s: int(s.rsplit("_", 1)[1]))
    test = sorted(ids[n_train:], key=lambda s: int(s.rsplit("_", 1)[1]))
    return {"seed": seed, "train_frac": train_frac, "category": CATEGORY,
            "n_train": len(train), "n_test": len(test), "train": train, "test": test}


def load_split():
    return json.loads(SPLIT_PATH.read_text())


def _sanitize(name):
    return re.sub(r"[-./:]", "_", name)


def cleanup_instances(uid):
    """BFCL keeps tool-environment instances in multi_turn_utils' module globals, keyed by
    model name + task id + class. Every episode uses a unique `uid` as model name; drop them."""
    g = vars(multi_turn_utils)
    prefix = _sanitize(uid) + "_"
    for k in [k for k in g if k.startswith(prefix)]:
        del g[k]


def system_prompt(entry):
    msgs = system_prompt_pre_processing_chat_model(
        copy.deepcopy(entry["question"][0]), entry["function"], entry["id"])
    assert msgs[0]["role"] == "system"
    return msgs[0]["content"]


def episode(entry, uid, parser="strict"):
    """Generator running one BFCL multi-turn episode.

    Yields `state` (dict) whenever the model must respond; the caller sends back the generated
    text. `state["messages"]` is the conversation so far, each message
    {"role": "user"|"assistant", "content": str, "turn": t, "kind": "query"|"call"|"tool_result"}.
    `state["turn"]` is the current turn index; the current user query is the last message with
    kind == "query". Returns (via StopIteration.value) the result dict.
    """
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]
    test_id = entry["id"]
    messages = []
    all_model_response = []
    force_quit = False
    n_calls = 0
    for turn_idx, turn_msgs in enumerate(entry["question"]):
        assert all(m["role"] == "user" for m in turn_msgs), turn_msgs
        # combine_consecutive_user_prompts: a turn's user messages become one message
        query = "\n\n".join(m["content"] for m in turn_msgs)
        messages.append({"role": "user", "content": query, "turn": turn_idx, "kind": "query"})
        turn_responses = []
        count = 0
        while True:
            text = yield {"messages": messages, "turn": turn_idx, "step": count}
            n_calls += 1
            messages.append({"role": "assistant", "content": text, "turn": turn_idx,
                             "kind": "call"})
            turn_responses.append(text)
            try:
                decoded = decode(text, parser)
                if is_empty_execute_response(decoded):
                    break
            except Exception:
                break
            results, _ = execute_multi_turn_func_call(
                decoded, initial_config, involved_classes, uid, test_id,
                long_context=False, is_evaL_run=False)
            tool_msg = format_execution_results_prompting(
                None, results, {"model_responses_decoded": decoded})
            messages.append({"role": "user", "content": tool_msg, "turn": turn_idx,
                             "kind": "tool_result"})
            count += 1
            if count > MAXIMUM_STEP_LIMIT:
                force_quit = True
                break
        all_model_response.append(turn_responses)
        if force_quit:
            break
    return {"model_response": all_model_response, "messages": messages,
            "force_quit": force_quit, "n_calls": n_calls}


def _decode_all(model_response, parser="strict"):
    decoded_turns = []
    for turn in model_response:
        dec = []
        for item in turn:
            try:
                d = decode(item, parser)
                if is_empty_execute_response(d):
                    continue
                dec.append(d)
            except Exception:
                continue
        decoded_turns.append(dec)
    return decoded_turns


def evaluate(entry, ground_truth, model_response, uid, parser="strict"):
    """Success exactly as BFCL's `_evaluate_single_multi_turn_entry`, plus a per-turn pass
    fraction (each turn checked like BFCL's checker, without stopping at the first failure)."""
    entry = copy.deepcopy(entry)
    out = {}
    if len(model_response) != len(ground_truth):
        out["valid"] = False
        out["error_type"] = "multi_turn:force_terminated"
    else:
        decoded = _decode_all(model_response, parser)
        res = multi_turn_checker(decoded, ground_truth, entry, CATEGORY, uid + "_chk")
        out["valid"] = bool(res["valid"])
        if not res["valid"]:
            out["error_type"] = res.get("error_type")
    cleanup_instances(uid + "_chk")
    out["turn_pass"] = per_turn_pass(entry, ground_truth, model_response, uid + "_pt", parser)
    cleanup_instances(uid + "_pt")
    n = len(ground_truth)
    out["turn_pass_frac"] = sum(out["turn_pass"]) / n if n else 0.0
    return out


def per_turn_pass(entry, ground_truth, model_response, uid, parser="strict"):
    """For each GT turn: pass iff the model's state after that turn matches GT state and the
    response check passes (same sub-checkers as BFCL). Turns after a force quit count as failed.
    Like BFCL, the model's own calls (not the GT calls) carry the state into later turns."""
    initial_config = entry["initial_config"]
    involved_classes = entry["involved_classes"]
    test_id = entry["id"]
    decoded = _decode_all(model_response, parser)
    passes = []
    all_model_results = []
    for t, gt_turn in enumerate(ground_truth):
        if t >= len(decoded):
            passes.append(0)
            continue
        model_instances = {}
        for step in decoded[t]:
            r, model_instances = execute_multi_turn_func_call(
                step, initial_config, involved_classes, uid, test_id, False, True)
            all_model_results.extend(r)
        gt_results, gt_instances = execute_multi_turn_func_call(
            gt_turn, initial_config, involved_classes, uid + "_gt", test_id, False, True)
        if not decoded[t] or is_empty_execute_response(decoded[t]):
            passes.append(0 if gt_turn else 1)
            continue
        if not gt_turn:
            passes.append(1)
            continue
        ok = state_checker(model_instances, gt_instances)["valid"] and \
            response_checker(all_model_results, gt_results, t)["valid"]
        passes.append(int(ok))
    return passes
