# ICAE + RL for a tool-using agent: feasibility report (Stage 0 + Stage 1)

Branch `icae-rl`, 2026-10-02 to 2026-10-05. Spec: `icae_rl/TASK_PROMPT.md` (verbatim). Code: `icae_rl/`.
Working notes: `icae_rl/NOTES.md`.

> **Not comparable to the public BFCL leaderboard.** Tasks are split 60/40 into TRAIN/TEST (RL would
> train on TRAIN), only the `multi_turn_base` category is used, and the agent prompt/decoding settings
> below are ours.

## 1. Summary

**The gate FAILS. Stage 2 (RL) was not started.**

- With the **full, uncompressed context**, Mistral-7B-Instruct-v0.2 (the model the released ICAE is
  built on) solves **1 of 120** TRAIN tasks (**0.8%**). The gate needs at least 15%.
- The main visible problem is the output format: only 14.5% of its calls can be parsed by BFCL. A
  diagnostic run with a forgiving parser (not BFCL's) raises that to 76.7%, but success only goes
  to **2/120 (1.7%)**. The model then loops and rambles, and almost never passes a turn after the
  first one. So the model is weak at the tasks themselves, not only at the format.
- Compression costs nothing measurable here, because there is almost nothing to lose. On the
  94 TRAIN tasks the (stopped) ICAE run reached, ICAE scored 1/94 with 9.0% of turns passed, and
  FULL scored 1/94 with 6.6%. So gate (b) also fails on the data we have.
- Conclusion: with this agent, RL on the compressor has no successes to learn from and no gap to
  close. The idea needs a stronger tool-using decoder, which means an ICAE trained on a different base
  model (section 7).

At the user's request (2026-10-02), the remaining Stage 1 runs were **stopped** and replaced by the
forgiving-parser diagnostic. Not measured: pretrained ICAE, "no history", the "compress earlier turns
only" variant, the full ICAE run (stopped at 94/120), and every TEST-split number.

## 2. Gate decision (TRAIN split)

| Condition | Required | Measured | Pass? |
|---|---|---|---|
| (a) FULL success rate | >= 15% | **0.8%** (1/120); forgiving-parser diagnostic 1.7% (2/120) | **No** |
| (b) FULL - ICAE | >= 5 points | ICAE 1/94 vs FULL 1/94 on the same 94 tasks: **0 points** (ICAE run stopped at 94/120) | **No** (on partial data) |
| (c) ICAE success > 0 | > 0 | 1/94 (task `multi_turn_base_50`, a 1-turn task that has no history to compress) | Yes, but only formally |

Gate (a) alone decides the outcome. Gate (c) passes only through a task where compression plays no
role. Partial-credit rewards (per-turn pass) would not rescue RL: per-turn pass is about 7-9% under
both FULL and ICAE, and the FULL and ICAE numbers are indistinguishable.

## 3. Setup

| Item | Value |
|---|---|
| Storage | `/data/a84460786/edgealign_icae_rl/` (base_models/, icae/, bfcl/, rollouts/, checkpoints/ (empty)) |
| Python env | `/data/a84460786/venvs/icae_rl`: torch 2.14.0+cu130, transformers 4.57.6, peft 0.21.2, numpy 1.26.4 |
| GPU | GPU 3 only (A100-40GB). GPU 0 is faulty (376 uncorrected ECC errors). The other GPUs were mostly in use by other users. |
| Benchmark | BFCL from github.com/ShishirPatil/gorilla @ `6ea5797` (2026-03-23; `bfcl_eval` installed editable, version string 0.0.0.dev0). Category `multi_turn_base`: 200 tasks. This category was introduced in BFCL v3 and is carried into the current V4 release unchanged except for ground-truth fixes (e.g. `multi_turn_base_154`, 2025-10-01). Data file: `BFCL_v4_multi_turn_base.json`. |
| Split | by task id, seed 0: TRAIN 120 / TEST 80 (`icae_rl/split.json`) |
| Decoding | greedy, max 512 new tokens per call (BFCL allows up to 4096), BFCL's step limit (more than 20 calls in a turn = force-terminated) |

**Agent format (prompting mode, as BFCL does for models without native function calling):**
- System prompt: BFCL's `system_prompt_pre_processing_chat_model` with the default format
  `ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic`, i.e. the
  function docs as indented JSON, and calls requested as
  `[func_name1(params_name1=params_value1, ...), func_name2(params)]`. It has 5.2k-11.7k tokens (mean
  8.5k).
- Mistral-7B-Instruct-v0.2's chat template has no system or tool role. The system prompt is merged
  into the first `[INST]`, as the official template and BFCL's `mistral_fc` handler do. Tool results
  are returned as one user message, `format_execution_results_prompting(...)`, as BFCL's prompting
  handlers do (e.g. `api_inference/mistral.py`). Assistant messages are the raw generated text.
- Calls are parsed with BFCL's `default_decode_execute_prompting`, and success is decided exactly as
  in BFCL's `_evaluate_single_multi_turn_entry` + `multi_turn_checker`. The per-turn pass fraction
  applies BFCL's state and response checkers to every turn without stopping at the first failure.
- Checks: the rendered prompt matches the official chat template. BFCL's checker accepts all 200
  ground-truth call lists. Replaying the ground truth *as model text* passes 176/200; all 24
  failures come from ground-truth calls with positional arguments, which BFCL's prompting parser
  drops (models are asked to write `name=value`).

**What gets compressed** (spec: keep the system prompt and the CURRENT user turn uncompressed):
at a model call in turn t, step k:
- `full`: P0 + earlier turns + current user query + this turn's earlier calls/tool results
- `icae` (literal reading, gated condition): P0 + `[slots(earlier turns)][FT]` + current query +
  `[slots(this turn's earlier calls/results)][FT]` + `[/INST]` (the final `[/INST]` stays as the
  generation cue). Slots sit where the text would have been.
- `icae_turn` (other reading of "current user turn": the turn's own calls stay uncompressed): planned,
  not run.
- `nohist`: P0 + current query: planned, not run.
- Pretrained ICAE: no FT token (its LM objective reads `[slots]` then continuation), and BOS at the
  start of the encoder input as in its pretraining. Fine-tuned ICAE: no BOS, as in its fine-tuning
  data code. Not run.
- Segments as in ICAE v2: `ceil(L/512)` segments of equal length, 128 slots each, each compressed on
  its own. Histories over 5120 tokens (ICAE's training maximum) are not truncated; they simply get
  more segments, which is the same multi-span mechanism. In the FULL runs, no strict-parser history
  exceeded 5120 tokens; 239 of 1366 calls did in the forgiving-parser run.

## 4. Stage 0: ICAE

- **Base checkpoint: Mistral-7B-Instruct-v0.2.** Both ICAE v2 inference scripts set
  `BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"`. The dataclass default `Mistral-7B-v0.1` is never
  used by the released scripts, and the fine-tuning prompt uses the Mistral instruct template token
  ids. We downloaded revision `63a8b08`, whose weight files have the same hashes as the March-2024
  revision `41b61a3` (released together with ICAE v2).
- **Weights:** `mistral_7b_ft_icae.safetensors` and `mistral_7b_pretrained_icae.safetensors`
  (safetensors only). Each holds LoRA r=512, alpha=32 on q_proj/v_proj of all 32 layers (218,103,808
  params) plus `memory_token_embed` [131 x 4096] (128 memory slots + AE/LM/FT tokens; 536,576 params).
- **Slots per input length:** 128 per segment of up to 512 tokens, i.e. `128 * ceil(L/512)`. A short
  input still gets 128 slots: 78 of 405 strict-run calls with history had more slots than history
  tokens.
- **Maximum input:** 5120 tokens (10 segments, 1280 slots) at training time.
- **Encoder/decoder separation:** one Mistral model. The encoder runs WITH the LoRA adapter (slots =
  final hidden states at the 128 memory-token positions). The decoder is the same weights with the
  adapter switched off (`with model.disable_adapter():`). At training time their code loads a second
  frozen copy only to allow gradient checkpointing.
- **Licenses:**

| Component | License |
|---|---|
| ICAE code | CC0 1.0 |
| ICAE weights | Apache-2.0 |
| Mistral-7B-Instruct-v0.2 | Apache-2.0 |
| BFCL / gorilla | Apache-2.0 |

- **Sanity check** (their scripts; only patch: sdpa instead of flash-attention-2):
  - The fine-tuned ICAE answers all 6 questions about their compressed news article correctly.
  - The pretrained ICAE reconstructs two prose texts exactly.
  - A short BFCL-style tool trace comes back lossy: two calls merged into one, a junk token, lost
    arguments.
  - Our re-implementation (`icae_rl/icae_model.py`, batched) reproduces their 6 outputs and the
    reconstruction token-for-token.
  - Outputs: `icae_rl/stage0/`.

## 5. Stage 1 results (TRAIN; TEST not run)

| Run | Tasks | Success | Turns passed | Episodes with >= 1 turn passed | Calls/episode | Force-terminated (more than 20 calls in a turn) | Calls BFCL can parse | Seconds/episode* |
|---|---|---|---|---|---|---|---|---|
| FULL (BFCL parser) | 120 | **1 (0.8%)** | 7.1% (32/449) | 31 | 4.4 | 0 | 14.5% | 26.6 (batch 6) |
| FULL, forgiving parser (diagnostic, not BFCL) | 120 | 2 (1.7%) | 12.5% (56/449) | 43 | 11.4 | 40 | 76.7%** | 156.6 (batch 4) |
| ICAE fine-tuned (BFCL parser), stopped | 94 of 120 | 1 (1.1%) | 9.0% | 26 | n/a*** | 12 (>20 calls) | n/a*** | 27.8 (batch 6) |
| FULL on the same 94 tasks | 94 | 1 (1.1%) | 6.6% | 22 | | | | |

\* Throughput: wall time / episodes with 4-6 episodes running concurrently on one A100.
\*\* parsed by the forgiving parser.
\*\*\* the stopped run's transcripts were not saved (only its log lines; episodes were written at
the end of a run at that time; fixed since).

**History length and slots** (what ICAE has to compress, measured on the FULL conversations):

| | Strict-parser FULL run | Forgiving-parser FULL run |
|---|---|---|
| History tokens per call (mean over all calls) | 315 | 2,622 |
| History tokens per call, calls with history | 408 (max 1,803) | 2,875 (max 18,732) |
| ICAE slots per call, calls with history | 174 | 801 |
| Decoder input, FULL (mean / max tokens) | 8,839 / 12,789 | 11,072 / 30,460 |
| System prompt (uncompressed, mean) | 8,486 | 8,486 |

The history is small next to the 8.5k-token system prompt that stays uncompressed, so compression
saves little context here even when it works.

**Where turns pass** (FULL): turn 0 passes 28/120 (strict) and 39/120 (forgiving). Later turns pass
only 4/329 and 17/329. The agent can sometimes do a first request, but it almost never carries a task
through later turns, which is exactly the part that depends on history.

**Why calls fail to parse** (strict FULL, 525 calls; `icae_rl/stage1/analysis.json`):

| Pattern | Share |
|---|---|
| parsed correctly | 12.6% |
| parsed, but positional args dropped | 1.9% |
| escaped underscores (`file\_name`, a markdown habit) | 29.9% |
| a call followed by explanation text | 19.0% |
| JSON-style calls (`{"name": ..., "parameters": ...}`) | 3.4% |
| prose, no call | 7.4% |
| other | 25.7% |

**The forgiving parser** (`bfcl_env.lenient_decode`, diagnostic only) un-escapes `\_`, strips code
fences and takes the first bracketed call list that BFCL's own decoder can parse. Execution and
success checking then go through BFCL unchanged. It does not loosen grading of correct calls: the
ground-truth replay result is the same, 176/200. Once more calls execute, the model:
- repeats the same call (40 episodes force-terminated),
- runs into the token cap (209 of 1366 responses hit 512 tokens),
- writes self-talk like "The given response is not a valid function call...".

## 6. Example episodes (abridged; more in `icae_rl/stage1/examples.md`)

**FULL, the only strict success (`multi_turn_base_50`, 1 turn):**
USER: "...all of my car doors seem to have locked themselves... help to get those doors unlocked..."
MODEL: `[lockDoors(unlock=True, door=["driver", "passenger", "rear_left", "rear_right"]), setHeadlights(mode="on")]`
TOOL: `{"lockStatus": "unlocked", ...}` -> MODEL: prose (ends the turn). BFCL: valid.

**FULL, typical strict failure (`multi_turn_base_1`):** every turn's call is followed by an
explanation, so BFCL's parser rejects it and the turn ends with nothing executed:
`[ls(a=true)]\n\nThis function call will list all the visible and hidden files ...`
`[cd("workspace")]\n[mv("log.txt", "archive/")]\n\nFirst, the current working directory ...`

**FULL, forgiving parser, same task:** the call now executes, but the model loops for 21 calls in
turn 0 and is force-terminated:
MODEL: `The given response is not a valid function call. Here's the correct format ... [ls(a="true")] ...`
TOOL: `{"current_directory_content": ["workspace"]}` (repeated 20 times).

**FULL, forgiving-parser success (`multi_turn_base_182`)**, which strict parsing loses only because
of `set\_budget\_limit(...)` in turn 1.

**ICAE fine-tuned (smoke test, `multi_turn_base_4`)**, an example of information lost in the slots.
In turn 0 the agent listed the directory, so FULL knows the file is `report.txt`. Under ICAE the
turn-0 history (45 tokens -> 128 slots) does not carry the name:
FULL  turn 1: `[{"name": "sort", "parameters": {"file\_name": "report.txt"}}]` (right file, wrong format)
ICAE  turn 1: `[sort(file_name="report")]` -> TOOL: `sort: report: No such file or directory`
ICAE  turn 2: `[sort('file.txt')]...` (the history no longer pins down the file)

## 7. What this means and possible next steps (none started)

1. **The decoder is the bottleneck, not the compressor.** Any test of "RL makes the compressor
   task-aware" needs an agent that succeeds with full context at a useful rate. Released ICAE weights
   exist only for Mistral-7B-Instruct-v0.2 and Llama-2-7b-chat, neither of which is a capable tool
   user.
2. **Option A, a stronger decoder plus our own ICAE:** pretrain and fine-tune ICAE (their
   `pretrain.py` / `instruction_finetune.py`) on a base with good function calling, e.g. a Qwen3
   model of 4-8B. Check its FULL score on this split first. This is a large job (ICAE pretraining),
   and it changes the project from "RL on the released ICAE" to "train ICAE, then RL".
3. **Option B, an easier environment** where Mistral-7B-Instruct-v0.2 succeeds with full context
   (e.g. a simpler multi-turn tool or QA task with automatic checks), so that the FULL-ICAE gap can be
   measured. Same RL design.
4. **Not recommended:** relaxing the parser or using partial-credit rewards to force Stage 2 on this
   setup. Even with both, FULL is at 1.7% / 12.5% per turn, and ICAE is not measurably worse, so RL
   would have no signal to recover.
5. If more Stage 1 numbers are wanted for completeness (TEST split, pretrained ICAE, no history,
   compress-earlier-turns-only), `icae_rl/run_stage1.py` runs them. It needs one free A100 for about
   6-7 h, and it is resumable per episode.

## 8. Differences from the spec

| Spec | What was done | Discussed with the user? |
|---|---|---|
| Measure all conditions on both splits | Only FULL (TRAIN) and a partial ICAE-ft run (94/120 TRAIN) | **Yes**: the user stopped the runs and asked for the forgiving-parser diagnostic instead |
| BFCL's parser | Additional diagnostic run with a forgiving parser (labelled; not used for the gate) | **Yes**: user request |
| "BFCL v3 multi-turn" | The same `multi_turn_base` category from the current V4 release (gorilla @ 6ea5797), with later ground-truth fixes | No: reported here |
| Run their inference script | Ran it with one patch (sdpa instead of flash-attention-2, which is not installed) | No: reported here |
| What is compressed | Literal reading: the current turn's earlier calls/tool results are compressed too; the other reading (`icae_turn`) was planned but not run | No: reported here |
| Max new tokens | 512 per call (BFCL allows up to 4096) | No: reported here |
| FT token after slots; BOS in the encoder input per adapter | Follows ICAE's own training formats | No: reported here |
| Seconds per episode | Throughput with 4-6 concurrent episodes, not single-episode latency | No: reported here |

## 9. Files

- Code: `icae_rl/bfcl_env.py` (BFCL episode and checks, strict/forgiving parser), `icae_rl/icae_model.py`
  (ICAE: compression + batched decoding), `icae_rl/rollout.py` (per-condition decoder inputs, batched
  episodes), `icae_rl/run_stage1.py`, `icae_rl/analyze_stage1.py`.
- Results: `icae_rl/stage1/summary.json`, `analysis.json`, `examples.md`; logs `icae_rl/logs/stage1*.out`.
- Full transcripts (outside git): `/data/a84460786/edgealign_icae_rl/rollouts/stage1/`.
