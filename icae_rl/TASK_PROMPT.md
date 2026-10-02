# TASK_PROMPT.md: the user's original spec for icae_rl (verbatim)

Pasted by the user on 2026-10-02. Agreed changes and deviations are listed in
`icae_rl/NOTES.md` (section "Deviations from the spec") and in the reports.

---

Create a new git branch named `icae-rl` from the current main, switch to it, and do
ALL work in this task on that branch. Never commit to or push main. Read AGENT.md,
MEMORY.md and SHORT_MEMORY.md first and follow the repo conventions (including
updating the memory files, on this branch). Put all new code in a new folder
icae_rl/. Do not modify existing files except the memory files.

GOAL
Test whether reinforcement learning can make a pretrained context compressor
(ICAE, In-context Autoencoder, Ge et al., ICLR 2024) task-aware for a tool-using
agent. The agent's growing context (earlier turns and tool outputs) is compressed
into ICAE memory slots that a FROZEN decoder LLM reads; RL updates ONLY the
compressor so the agent succeeds more often. Three conditions are compared
throughout:
  (1) FULL: the agent sees its full, uncompressed context (upper bound).
  (2) ICAE: the history is compressed by the released ICAE, no RL.
  (3) ICAE+RL: the history is compressed by the RL-trained compressor.
The hypothesis: (3) recovers much of the gap between (2) and (1).

This work has two stages with a GATE between them. Do Stage 1 first. Only start
Stage 2 if the gate passes. If it fails, write the feasibility report, push, and
stop.

STORAGE
All large files go under /data/, never in git. Use
/data/a84460786/edgealign_icae_rl/ (create it), with subfolders:
  base_models/   downloaded base LLM(s)
  icae/          the released ICAE weights
  checkpoints/   all RL checkpoints
  rollouts/      saved episodes / generations, if large
If that path is not writable, pick another location under /data/, say which in
the notes, and use it consistently. Add any local paths to .gitignore as needed.

STAGE 0 - SET UP ICAE
1. Clone github.com/getao/icae (into /data/.../icae/code or a scratch dir, not
   into this repo). Read code/icae_v2: the model definition, inference scripts,
   pretrain.py, instruction_finetune.py and training_utils.py.
2. Download the released Mistral-7B ICAE weights from the Hugging Face repo
   sggetao/icae: mistral_7b_pretrained_icae.safetensors and
   mistral_7b_ft_icae.safetensors. Use the .safetensors files only; do not load
   any pickled .pt files.
3. From their code, determine EXACTLY which Mistral-7B base checkpoint ICAE v2
   was trained on (base or instruct, which version) and download that one. Report
   it. Also report: the number of memory slots per input length, the maximum
   input length, how the LoRA encoder and frozen decoder are separated (how the
   adapter is switched off for decoding), and the license of each component.
4. Sanity check: run their inference script with the fine-tuned model on a few
   examples (compress a paragraph, ask a question about it), and with the
   pretrained model test reconstruction of a short text. Report the outputs.
   Stop and report if these don't behave sensibly.

STAGE 1 - FEASIBILITY (no training)
Benchmark: use the multi-turn "base" category of the Berkeley Function Calling
Leaderboard (BFCL v3 multi-turn), which runs simulated tools locally and checks
success automatically from the final environment state and responses.
- Install it from the official gorilla repository (berkeley-function-call-
  leaderboard) and run it locally. Report the version/commit.
- Mistral-7B may lack native function calling: use BFCL's prompting-based
  function-calling format for models without native tool support (follow how
  BFCL handles such models), and report exactly which format you used.
- Split the multi-turn base tasks by task id into TRAIN (60%) and TEST (40%),
  fixed seed 0, and save the split. RL will train on TRAIN only. Note in every
  report that because RL trains on part of BFCL, these scores are not comparable
  to the public leaderboard.
- If BFCL multi-turn cannot be run locally, or the task count is too small to
  split usefully, report why and propose an alternative environment with
  automatic success checks; do not switch silently.

What gets compressed: at every turn, the agent keeps UNCOMPRESSED the system
prompt with the function documentation and the CURRENT user turn. Everything
earlier (previous user turns, the agent's previous tool calls and responses, and
all tool outputs) is compressed by ICAE into memory slots, which are placed where
that history would have been. Follow ICAE's own limits on input length and slot
count; if the history exceeds the maximum, compress it in segments as ICAE v2
supports (multi-span), and report how.

Measure on BOTH splits, with greedy decoding:
- FULL success rate,
- ICAE success rate (fine-tuned ICAE model; also report the pretrained one),
- the average history length in tokens and the average number of slots,
- seconds per episode for each condition.
Also run a "no history" condition (history dropped entirely, nothing inserted),
to show how much the history matters at all.

GATE (decide using the TRAIN split, and report the numbers):
Proceed to Stage 2 only if ALL hold:
  (a) FULL success rate >= 15% (the agent can do the task at all);
  (b) FULL - ICAE >= 5 percentage points (compression costs something, so RL has
      room to improve);
  (c) ICAE success rate > 0 on TRAIN (RL needs some successes to learn from;
      if it is 0 but (a) and (b) hold, report it and propose partial-credit
      rewards instead of starting RL).
Write icae_rl_feasibility_report.md (setup, everything from Stage 0, all
numbers per condition and split, gate decision with reasons, example episodes
for each condition) and commit + push the branch. If the gate fails, stop here.

STAGE 2 - RL ON THE COMPRESSOR
Trainable: ONLY the ICAE encoder's LoRA adapter and the memory token embeddings.
The decoder (base Mistral with the adapter switched off) stays frozen. Start from
the fine-tuned ICAE weights. Assert at startup that no decoder parameter requires
grad and that the trainable parameter count matches the LoRA + memory tokens.

Algorithm (GRPO-style policy gradient):
- Each RL step: sample 8 TRAIN tasks; for each, run G = 4 full episodes with the
  CURRENT compressor, decoder sampling at temperature 0.7, top_p 0.95.
- Reward per episode: 1 if BFCL marks the task successful, else 0. Also log
  BFCL's per-turn pass fraction; if the gate's condition (c) was borderline,
  use reward = per-turn pass fraction instead, and say so.
- Advantage: reward minus the mean reward of that task's G episodes, divided by
  their std + 1e-6; tasks whose G episodes all got the same reward contribute
  nothing (skip them, and log how many were skipped).
- Loss: -advantage x (sum of log-probs of all tokens the decoder generated in
  the episode), with log-probs recomputed in a teacher-forced forward pass after
  sampling, so the gradient flows decoder -> memory slots -> encoder LoRA.
  Recompute the memory slots inside that pass, with gradients. Normalize by the
  total number of generated tokens in the step.
- Add a KL penalty, beta = 0.02, between the current policy and the initial
  (Stage-0 fine-tuned ICAE) policy on the generated tokens.
- AdamW, LR 1e-5, betas (0.9, 0.95), eps 1e-6, grad clip 1.0, bf16 with
  gradient checkpointing. 200 RL steps.
- Save the sampled episodes of every step (compressed form is fine) under
  /data/.../rollouts/.

Evaluation every 20 RL steps, on the TEST split, greedy decoding:
- ICAE+RL success rate, next to the fixed FULL and ICAE numbers from Stage 1;
- mean reward on TRAIN from the last 20 steps;
- KL to the initial policy;
- COLLAPSE CHECKS (earlier runs in this repo collapsed silently, see
  gist_health_report.md and warmstart_collapse_report.md):
    * mean pairwise cosine between memory slots of different tasks
      (alert if > 0.95),
    * swap test: run 32 TEST tasks with memory slots taken from a different task;
      success should drop clearly (alert if the swapped success is within 2
      points of the own-slot success);
    * reconstruction check with the pretrained-ICAE decoding prompt on 16
      histories, to see whether the slots still encode content.
  Log a clear WARNING if any alert fires; do not stop automatically.
Checkpoints under /data/.../checkpoints/: latest (every eval), best by TEST
success, and one every 50 steps. Support --resume. Log per-step metrics to
icae_rl/logs/train.jsonl and evals to icae_rl/logs/eval.jsonl, and regenerate
icae_rl/success_curve.png at every eval (TEST success of ICAE+RL vs step, with
horizontal lines for FULL and ICAE, and mean TRAIN reward).

RUNNING
- Check nvidia-smi first and use healthy, free GPUs (GPU 0 had a hardware error
  before). Report which GPUs. If two GPUs are used, keep the update identical to
  a single-GPU one, as in the warm-start trainer.
- Before the full RL run: a 2-step smoke run; report seconds per step and the
  estimated total time. If the estimate exceeds 48 hours, reduce the number of
  RL steps and say so before launching.
- Launch the full run with nohup, logging to icae_rl/logs/rl.out.

GIT (branch icae-rl only)
- Commit + push after Stage 0, after Stage 1 (with the feasibility report), at
  RL step 40, and at the end.
- At the end, write icae_rl_report.md: plain-language summary; the three-way
  comparison (FULL / ICAE / ICAE+RL) on TEST with the best step; the training
  curve; collapse-check results over time; example episodes where RL changed the
  outcome; what went wrong or looks suspicious; recommended next steps. Do not
  overclaim; if results are mixed, say so.
- Confirm every push succeeded, and update the memory files on the branch.
