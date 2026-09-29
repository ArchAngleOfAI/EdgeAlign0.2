"""Test 3 (Hypothesis A): is the student's LM_head really untied from Embedding1, and does it stay untied?

1. Build via build_softprompt_generator and compare data_ptr() of lm_head.weight,
   embed_tokens.weight (Embedding1) and embedding2.weight.
2. Run one real optimizer step (same loop as distill_softprompt.py) and confirm
   the pointers are still distinct, lm_head moved, and Embedding1 did not.
3. Statically search this repo and the installed transformers package for code
   that could silently re-tie (tie_weights / resize_token_embeddings /
   from_pretrained / save_pretrained / .generate) and report every call site.
4. Show what WOULD happen if tie_weights() were called on the built model
   (config.tie_word_embeddings is still True), on a throwaway copy at the end.
"""

import os
import re

import torch
import transformers

from _common import (
    LR, REPO_ROOT, forward_loss, get_device, get_tokenizer, load_all, setup_stdout, take_batches,
    trainable_params,
)

PATTERN = re.compile(r"tie_weights\(|resize_token_embeddings\(|save_pretrained\(|\.generate\(|from_pretrained\(")


def ptrs(encoder, embedding2):
    return {"lm_head": encoder.lm_head.weight.data_ptr(),
            "embed_tokens (Embedding1)": encoder.model.embed_tokens.weight.data_ptr(),
            "embedding2": embedding2.weight.data_ptr()}


def report_ptrs(tag, encoder, embedding2):
    p = ptrs(encoder, embedding2)
    print(f"[{tag}] data_ptrs: " + ", ".join(f"{k}={v:#x}" for k, v in p.items()))
    print(f"[{tag}] lm_head IS embed_tokens (same Parameter object): "
          f"{encoder.lm_head.weight is encoder.model.embed_tokens.weight}; "
          f"same data_ptr: {p['lm_head'] == p['embed_tokens (Embedding1)']}; "
          f"all three distinct: {len(set(p.values())) == 3}")


def search(root, files_filter=None, label=""):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv", "diagnostics"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            if files_filter and not files_filter(path):
                continue
            with open(path, errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if PATTERN.search(line) and not line.lstrip().startswith("#"):
                        hits.append((os.path.relpath(path, root), i, line.strip()))
    print(f"\n{label}: {len(hits)} call sites")
    for path, i, line in hits:
        print(f"  {path}:{i}: {line[:120]}")
    return hits


def enclosing_defs(path, needle):
    """Which functions in `path` contain a line matching `needle`."""
    out, current = [], None
    with open(path) as f:
        for i, line in enumerate(f, 1):
            m = re.match(r"\s*def (\w+)", line)
            if m:
                current = m.group(1)
            if needle in line and "def tie_weights" not in line:
                out.append((i, current, line.strip()))
    return out


def main():
    setup_stdout()
    device = get_device()
    encoder, embedding2, w1, teacher, receiver = load_all(device)

    print(f"config.tie_word_embeddings = {encoder.config.tie_word_embeddings}")
    report_ptrs("after build", encoder, embedding2)
    print(f"w1 (captured before untying) is embed_tokens: {w1 is encoder.model.embed_tokens.weight}")
    print(f"values equal at init: lm_head==Emb1 {torch.equal(encoder.lm_head.weight, w1)}, "
          f"embedding2==Emb1 {torch.equal(embedding2.weight, w1)}")

    emb1_before = encoder.model.embed_tokens.weight.detach().clone()
    params = trainable_params(encoder, embedding2)
    lm_before = encoder.lm_head.weight.detach().clone()
    opt = torch.optim.AdamW(params, lr=LR)
    prompt_ids = take_batches(get_tokenizer(), 1)[0].to(device)
    opt.zero_grad()
    forward_loss(encoder, embedding2, teacher, receiver, prompt_ids).backward()
    opt.step()
    report_ptrs("after 1 optimizer step", encoder, embedding2)
    print(f"lm_head changed: {(encoder.lm_head.weight - lm_before).norm().item():.4e}  "
          f"Embedding1 changed: {(encoder.model.embed_tokens.weight - emb1_before).norm().item():.4e}")
    print(f"encoder.train()/eval()/.to() re-tie?  ", end="")
    encoder.train(); encoder.eval(); encoder.to(device)
    print(encoder.lm_head.weight is encoder.model.embed_tokens.weight)

    search(REPO_ROOT, label="Repo call sites (tie_weights/resize/save_pretrained/generate/from_pretrained)")

    tf_root = os.path.dirname(transformers.__file__)
    mu = os.path.join(tf_root, "modeling_utils.py")
    print(f"\ntransformers {transformers.__version__}: functions in modeling_utils.py that call tie_weights():")
    for i, fn, line in enclosing_defs(mu, "tie_weights("):
        print(f"  modeling_utils.py:{i} inside {fn}(): {line}")
    gen_hits = [h for h in enclosing_defs(os.path.join(tf_root, "generation", "utils.py"), "tie_weights(")]
    print(f"generation/utils.py tie_weights() calls: {len(gen_hits)}")
    q3 = os.path.join(tf_root, "models", "qwen3", "modeling_qwen3.py")
    print(f"modeling_qwen3.py tie_weights() calls: {len(enclosing_defs(q3, 'tie_weights('))}; "
          f"_tied_weights_keys: {getattr(type(encoder), '_tied_weights_keys', None)}")

    # Latent risk demo: since config still says tie_word_embeddings=True, an explicit tie_weights()
    # (which resize_token_embeddings / init_weights would also trigger) re-ties them.
    encoder.tie_weights()
    print(f"\nAFTER an explicit encoder.tie_weights(): lm_head IS embed_tokens -> "
          f"{encoder.lm_head.weight is encoder.model.embed_tokens.weight}  "
          f"(demonstration only; nothing in the training path calls this)")


if __name__ == "__main__":
    main()
