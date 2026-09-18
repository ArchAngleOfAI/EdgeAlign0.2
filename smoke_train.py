"""Smoke test: 10 training steps on Qwen3-0.6B architecture.

By default loads only the model's config/architecture from the local
checkpoint dir (no pretrained weights, random init). With --pretrained,
loads the actual pretrained weights from the same checkpoint instead,
everything else (seed, hyperparameters) held identical, so the two runs
are a direct random-init-vs-pretrained comparison.

--data selects the input data:
- synthetic (default): a single fixed random-token batch (seed 0),
  reused across all 10 steps, to isolate loop mechanics from data.
- fineweb-edu: real text from HuggingFaceFW/fineweb-edu (the "sample-10BT"
  subset, its smallest sample split), packed into seq_len-token chunks
  with the model's own tokenizer, EOS-separated between documents. A
  FRESH chunk is used each step (this is a real dataloader, not a
  reused batch) — so the loss curve here is a genuine (if tiny) training
  curve, not the same single-batch-overfit signal as the synthetic mode.

fineweb-edu loading note (a deliberate implementation choice, flagged
since it deviates from the obvious `datasets.load_dataset(streaming=True)`
call): that path was tested and turned out to download the *entire*
~2GB parquet shard before yielding a single row, since HF's default
streaming reader doesn't do row-group-level lazy reads for this dataset.
Instead we open the first sample-10BT parquet file directly via fsspec
and read only its first row group (1000 docs, via HTTP range requests) —
same underlying HF-hosted data, but avoids the multi-minute download for
what only needs a few thousand tokens.

Assumptions made (not specified by the user — flagged here, adjust if wrong):
- dtype: float32 (not the checkpoint's declared bf16), to isolate the
  training-loop mechanics from any bf16 numerical issues.
- Fixed seed (0) for reproducibility (model init in random-init mode,
  and the synthetic data).
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/models/huggingface/qwen3-0.6b"
FINEWEB_EDU_SHARD_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/"
    "sample/10BT/000_00000.parquet"
)
SEQ_LEN = 1024
BATCH_SIZE = 1
LR = 5e-5
NUM_STEPS = 10
SEED = 0


def fineweb_edu_batches(tokenizer, seq_len, batch_size):
    """Yield fresh (batch_size, seq_len) token-id batches of real FineWeb-Edu text.

    Reads row groups from the first sample-10BT parquet shard lazily via
    HTTP range requests (no full-file download), tokenizes each document,
    concatenates with an EOS separator, and slices off fixed-length chunks
    as soon as enough tokens have accumulated. Runs indefinitely; the
    caller decides how many batches to pull (e.g. via itertools.islice).
    """
    import fsspec
    import pyarrow.parquet as pq

    eos_id = tokenizer.eos_token_id
    chunk_size = seq_len * batch_size

    parquet_file = pq.ParquetFile(fsspec.open(FINEWEB_EDU_SHARD_URL, "rb").open())
    buffer = []
    for row_group_idx in range(parquet_file.num_row_groups):
        texts = parquet_file.read_row_group(row_group_idx, columns=["text"]).column("text")
        for text in texts:
            buffer.extend(tokenizer(str(text), add_special_tokens=False)["input_ids"])
            buffer.append(eos_id)
            while len(buffer) >= chunk_size:
                chunk, buffer = buffer[:chunk_size], buffer[chunk_size:]
                yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true",
                         help="Load pretrained weights instead of random init.")
    parser.add_argument("--data", choices=["synthetic", "fineweb-edu"], default="synthetic",
                         help="Input data source (default: synthetic).")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(MODEL_PATH)
    if args.pretrained:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    if args.data == "fineweb-edu":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        batch_iter = fineweb_edu_batches(tokenizer, SEQ_LEN, BATCH_SIZE)
    else:
        fixed_batch = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN))
        batch_iter = (fixed_batch for _ in iter(int, 1))  # same batch forever

    print(f"mode={'pretrained' if args.pretrained else 'random-init'}, data={args.data}, "
          f"device={device}, vocab_size={config.vocab_size}, "
          f"seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, NUM_STEPS + 1):
        input_ids = next(batch_iter).to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"step {step:2d}/{NUM_STEPS}  loss={loss.item():.6f}")

    print("done.")

if __name__ == "__main__":
    main()
