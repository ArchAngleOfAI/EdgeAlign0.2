"""Load pretrained Qwen3-0.6B weights and compute W_LM^T @ W_emb.

W_emb = embed_tokens.weight, shape (vocab_size=151936, hidden_size=1024).
W_LM  = lm_head.weight,      shape (vocab_size=151936, hidden_size=1024).
(These are tied in this checkpoint -- literally the same tensor -- so
the result is the (1024, 1024) Gram matrix W_emb^T @ W_emb, computed
directly from the loaded tensors rather than assumed.)

Saves a heatmap of the resulting 1024x1024 matrix to a PNG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "/data/models/huggingface/qwen3-0.6b"
OUT_PNG = "/home/a84460786/testfolder/wlmT_wemb_heatmap.png"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.to(device)

    w_emb = model.model.embed_tokens.weight   # (vocab_size, hidden_size)
    w_lm = model.lm_head.weight                # (vocab_size, hidden_size)
    print("W_emb shape:", tuple(w_emb.shape))
    print("W_LM  shape:", tuple(w_lm.shape))
    print("tied (same tensor):", w_emb.data_ptr() == w_lm.data_ptr())

    with torch.no_grad():
        result = w_lm.T @ w_emb  # (hidden_size, hidden_size)
    print("result shape:", tuple(result.shape))

    result_np = result.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(result_np, cmap="RdBu_r", vmin=-abs(result_np).max(), vmax=abs(result_np).max())
    ax.set_title(r"$W_{LM}^T W_{emb}$ — Qwen3-0.6B (tied embedding/LM-head)")
    ax.set_xlabel("hidden dim (W_emb column)")
    ax.set_ylabel("hidden dim (W_LM^T row)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print("saved:", OUT_PNG)

if __name__ == "__main__":
    main()
