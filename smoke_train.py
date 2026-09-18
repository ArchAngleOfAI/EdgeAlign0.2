"""Smoke test: 10 training steps on Qwen3-0.6B architecture.

By default loads only the model's config/architecture from the local
checkpoint dir (no pretrained weights, random init). With --pretrained,
loads the actual pretrained weights from the same checkpoint instead,
everything else (seed, synthetic batch, hyperparameters) held identical,
so the two runs are a direct random-init-vs-pretrained comparison.

Assumptions made (not specified by the user — flagged here, adjust if wrong):
- dtype: float32 (not the checkpoint's declared bf16), to isolate the
  training-loop mechanics from any bf16 numerical issues.
- Same fixed random batch is reused across all 10 steps (rather than a
  fresh random batch each step), so a decreasing loss is a meaningful
  signal that backprop/optimizer are actually working.
- Fixed seed (0) for reproducibility, and used for both the model init
  (random-init case) and the synthetic data generation in both cases.
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_PATH = "/data/models/huggingface/qwen3-0.6b"
SEQ_LEN = 1024
BATCH_SIZE = 1
LR = 5e-5
NUM_STEPS = 10
SEED = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true",
                         help="Load pretrained weights instead of random init.")
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

    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = input_ids.clone()

    print(f"mode={'pretrained' if args.pretrained else 'random-init'}, "
          f"device={device}, vocab_size={config.vocab_size}, "
          f"seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"step {step:2d}/{NUM_STEPS}  loss={loss.item():.6f}")

    print("done.")

if __name__ == "__main__":
    main()
