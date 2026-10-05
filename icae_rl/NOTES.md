# icae_rl working notes

Running notes for the ICAE + RL task (`TASK_PROMPT.md`). The reports
(`icae_rl_feasibility_report.md`, later `icae_rl_report.md`) are written from these notes.

## Storage and environment

- Large files: `/data/a84460786/edgealign_icae_rl/` (writable, used as specified):
  `base_models/Mistral-7B-Instruct-v0.2/` (revision 63a8b08, see below), `icae/` (released
  ICAE safetensors + `code/` = clone of github.com/getao/icae @ 469a468, 2024-05-11;
  `sanity/` = the patched copy used for the Stage 0 sanity run), `bfcl/gorilla/` (BFCL clone),
  `checkpoints/`, `rollouts/`.
- Python env: `/data/a84460786/venvs/icae_rl` (new, so the gist env stays untouched):
  Python 3.12.3, torch 2.14.0+cu130, transformers 4.57.6, peft 0.21.2, numpy 1.26.4 (pinned by
  BFCL), bfcl_eval installed editable from the clone.
- GPUs at start (2026-10-02 13:18 UTC): GPU 3 free; 0 faulty (376 uncorrected ECC errors, the
  known hardware error) and running someone's vLLM worker; 2 has another user's vLLM worker at
  100% util (~39 GB free); 1, 4, 5, 6, 7 mostly full (other users). Stage 0 ran on GPU 3.

## Stage 0: ICAE

### Base checkpoint
- **Mistral-7B-Instruct-v0.2** (instruct, v0.2). Evidence: both v2 inference scripts
  (`fine_tuned_inference_script.sh`, `pretrained_inference_script.sh`) set
  `BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"` (the v0.1 base and Llama lines are commented
  out). The dataclass default in `modeling_icae_multi_span.py` (`mistralai/Mistral-7B-v0.1`) is
  never used by the released scripts. The fine-tuning prompt uses the Mistral instruct template
  (ids `[1, 733, 16289, 28793]` = `<s>▁[INST]`, `[733, 28748, 16289, 28793]` = `▁[/INST]`).
  The sanity outputs below confirm the match (exact reconstruction, correct QA).
- Downloaded revision 63a8b081 (2025-07-24, README-only change). Weight files have the same LFS
  hashes as revision 41b61a33 (2024-03-24, when ICAE v2 was released), so the weights are
  byte-identical to what ICAE v2 used. Only the tokenizer/chat template were updated in between,
  and ICAE hard-codes its template token ids. safetensors only; no .pt / .bin loaded.

### Architecture facts (from `code/icae_v2`)
- **One** Mistral model wrapped by PEFT LoRA: r=512, alpha=32 (scaling 1/16), dropout 0.05,
  default target modules for Mistral = `q_proj`, `v_proj`, all 32 layers: 218,103,808 params.
  Plus `memory_token_embed` = nn.Embedding(128 + 3, 4096) fp32 (rows 0-127 memory tokens, 128 =
  AE token, 129 = LM token, 130 = FT token): 536,576 params. Released file: 218,640,384 params
  in total (129 tensors); same layout for the pretrained and the fine-tuned file.
- **Encoder** = the model WITH the adapter. Input: one segment of text tokens followed by the
  128 memory-token ids (32001..32128, embedded by `memory_token_embed`); the memory slots are
  the final-layer hidden states (`hidden_states[-1]`, after the final RMSNorm) at those 128
  positions.
- **Decoder** = the same base weights with the adapter switched off:
  `with self.icae.disable_adapter(): self.icae(inputs_embeds=...)`. At training time their code
  loads a second frozen copy (`self.decoder`) only to allow gradient checkpointing; at inference
  it is the PEFT context manager. Memory slots are fed as `inputs_embeds`; special tokens (AE/FT)
  are embedded by `memory_token_embed`.
- **Slots per input length:** fixed 128 slots per segment; segment budget = mem 128 x mean
  compression rate 4 = 512 tokens. `num_segments = ceil(L / 512)`, segments of equal length
  `ceil(L / num_segments)`, each compressed **independently** (no cross-segment context), slots
  concatenated: **slots = 128 * ceil(L / 512)**. A 30-token input still gets 128 slots (the
  "compressed" form can be longer than the text for short inputs).
- **Maximum input length:** 5120 tokens (`--model_max_length 5120`, inputs truncated at 5120
  in the inference scripts and in instruction fine-tuning), i.e. at most 10 segments / 1280
  slots at train time. This multi-segment ("multi-span") mechanism is what v2 adds.
- Prompt formats: pretrained (autoencoding) = `[slots][AE]` -> text + EOS (input tokenized WITH
  BOS); LM continuation = `[slots]` -> continuation; fine-tuned =
  `<s>[INST][slots][FT]{prompt} [/INST]` -> answer (input tokenized WITHOUT BOS in
  `instruct_ft_tokenize_function`, though their inference script adds BOS).
- Training precision: bf16 required (README warns against fp16). Batch size 1 only in v2 code.

### Licenses
| Component | License | Source |
|---|---|---|
| ICAE code (github.com/getao/icae) | CC0 1.0 Universal | repo `LICENSE` |
| ICAE weights (huggingface sggetao/icae) | Apache-2.0 | model card metadata |
| Mistral-7B-Instruct-v0.2 | Apache-2.0 | model card metadata |
| BFCL / gorilla (code + data) | Apache-2.0 | repo `LICENSE`, pyproject |

### Sanity check (GPU 3, their scripts; only patch: `use_flash_attention_2=True` ->
`attn_implementation="sdpa"` since flash-attn is not installed; local base path)
Fine-tuned ICAE on their `dev_v2.jsonl` (one 2,350-char news article, 6 questions; greedy):
| Question | Gold | ICAE output |
|---|---|---|
| Identify the person arrested ... | Benoit Quennedey | Benoit Quennedey |
| List the actions taken against ... | His office ... was raided by DGSI officers. | ... detained on Sunday morning and his office in the French Senate was raided by the DGSI. |
| What is the role of ... | senior civil servant who liaises ... | civil servant who liaises between the French Senate and the Department of Architecture and Heritage ... |
| What are the charges ... | collecting and delivering to a foreign power information ... | charged with collecting and delivering information likely to subvert core national interests |
| Mention the organization ... | Franco-Korean Friendship Association | Franco-Korean Friendship Association |
| When did the ... investigation begin? | In March of this year. | March |

Pretrained ICAE reconstruction (`[slots][AE]`, greedy):
- their condiment paragraph (~70 tokens): reconstructed **exactly** (plus a leading `<s>`,
  because the input includes BOS).
- "The quick brown fox ... third cup of coffee.": **exact**.
- a short BFCL-style agent trace (user turn + two tool calls + tool results, ~150 tokens):
  **lossy**: the two calls are merged into `cd(folder='documents', mv='final_report.pdf',
  destination='temp')`, a junk token `ivals` appears, the `mv` call loses its arguments, and the
  output ends with a stray `[`. Early sign that the released ICAE is weaker on tool traces than
  on prose.
Raw outputs: `icae_rl/stage0/`.

## Stage 1 (summary; details in icae_rl_feasibility_report.md)

- FULL TRAIN strict 1/120; forgiving parser 2/120; ICAE-ft stopped at 94/120 (1/94). Gate failed.
- The user stopped the remaining Stage 1 runs on 2026-10-02 and asked for the forgiving-parser
  diagnostic instead. Lenient run OOM'd twice before the generate() memory fix (logs/stage1_lenient_oom_*).
