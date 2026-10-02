"""ICAE (Mistral-7B v2) re-implemented for icae_rl: compression + batched decoding.

Same math as getao/icae `code/icae_v2/modeling_icae_multi_span.py`:
  - one Mistral-7B-Instruct-v0.2 with a PEFT LoRA adapter (r=512, alpha=32, q_proj/v_proj);
  - encoder = model WITH the adapter: [segment tokens][128 memory tokens] -> final hidden states
    (after the final norm) at the memory positions = 128 slots per segment;
  - segments: num = ceil(L / 512), length ceil(L / num), compressed independently;
  - decoder = the same weights with the adapter disabled (`disable_adapter()`); slots and the
    special tokens (AE / LM / FT, rows 128..130 of memory_token_embed) enter as inputs_embeds.
Differences from their code (no effect on the math): no vocabulary resize (ids >= 32000 never
reach embed_tokens; their logits are sliced to the first 32000 anyway), sdpa attention instead of
flash-attention-2, batched/left-padded encoder and decoder calls with explicit position ids,
LoRA dropout 0 (their inference runs in eval mode, so dropout is off there too).
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path("/data/a84460786/edgealign_icae_rl")
BASE_PATH = ROOT / "base_models" / "Mistral-7B-Instruct-v0.2"
ICAE_PATHS = {
    "ft": ROOT / "icae" / "mistral_7b_ft_icae.safetensors",
    "pre": ROOT / "icae" / "mistral_7b_pretrained_icae.safetensors",
}
MEM = 128                 # fixed_mem_size
COMPRESSION_RATE = 4      # mean_compression_rate
SEG = MEM * COMPRESSION_RATE   # 512 tokens per segment
MAX_INPUT = 5120          # model_max_length used to train ICAE v2
AE_ROW, LM_ROW, FT_ROW = MEM, MEM + 1, MEM + 2
VOCAB = 32000
BOS, EOS = 1, 2
INST_OPEN = [733, 16289, 28793]          # "▁[INST]"
INST_CLOSE = [733, 28748, 16289, 28793]  # "▁[/INST]"
HIDDEN = 4096


def lora_config():
    return LoraConfig(r=512, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])


def segment(ids):
    """ICAE v2 multi-span split: list of segments (each <= 512 tokens)."""
    L = len(ids)
    if L == 0:
        return []
    n = math.ceil(L / SEG)
    seg_len = math.ceil(L / n)
    return [ids[i * seg_len:(i + 1) * seg_len] for i in range(n)]


def n_slots(n_tokens):
    return MEM * math.ceil(n_tokens / SEG) if n_tokens > 0 else 0


class ICAE(nn.Module):
    def __init__(self, device, adapters=None):
        """adapters: {adapter_name: source} with source in ICAE_PATHS ("ft" / "pre")."""
        super().__init__()
        adapters = adapters or {"ft": "ft"}
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(BASE_PATH)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_PATH, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        names = list(adapters)
        self.peft = get_peft_model(base, lora_config(), adapter_name=names[0])
        for n in names[1:]:
            self.peft.add_adapter(n, lora_config())
        self.mem = nn.ModuleDict({n: nn.Embedding(MEM + 3, HIDDEN) for n in names}).to(device)
        for n, src in adapters.items():
            self._load(n, ICAE_PATHS[src])
        for p in self.parameters():
            p.requires_grad_(False)
        self.trainable = []   # parameters kept trainable across adapter switches
        self.inner = base.model          # MistralModel (LoRA layers injected in place)
        self.lm_head = base.lm_head
        self.embed = self.inner.embed_tokens
        self.active = None
        self.eval()

    # ---------------------------------------------------------------- weights
    def _load(self, name, path):
        sd = load_file(str(path))
        params = dict(self.peft.named_parameters())
        n_lora = 0
        for k, v in sd.items():
            if k == "memory_token_embed.weight":
                self.mem[name].weight.data.copy_(v.float())
                continue
            assert k.startswith("icae.") and ".default." in k, k
            tk = k[len("icae."):].replace(".default.", f".{name}.")
            params[tk].data.copy_(v.to(params[tk].dtype))
            n_lora += 1
        assert n_lora == 128, n_lora   # 32 layers x {q,v} x {A,B}

    def lora_params(self, name):
        return [p for n, p in self.peft.named_parameters() if f".{name}." in n and "lora_" in n]

    def use(self, name):
        """Activate an adapter for encoding. peft's set_adapter flips requires_grad, so the
        trainable set is restored afterwards."""
        if self.active != name:
            self.peft.set_adapter(name)
            self.active = name
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.trainable:
            p.requires_grad_(True)

    # ---------------------------------------------------------------- text helpers
    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def special(self, adapter, row):
        return self.mem[adapter].weight[row].to(torch.bfloat16)

    # ---------------------------------------------------------------- encoder
    def compress(self, histories, adapter, max_batch_segments=12):
        """histories: list of token-id lists. Returns a list of [n_slots, 4096] bf16 tensors
        (empty tensor for an empty history). Differentiable w.r.t. the active adapter's LoRA and
        memory embeddings when grad is enabled."""
        self.use(adapter)
        segs, owner = [], []
        for h, ids in enumerate(histories):
            for s in segment(ids):
                segs.append(s)
                owner.append(h)
        outs = []
        memw = self.mem[adapter].weight[:MEM].to(torch.bfloat16)
        for b0 in range(0, len(segs), max_batch_segments):
            chunk = segs[b0:b0 + max_batch_segments]
            T = max(len(s) for s in chunk) + MEM
            embs, masks = [], []
            for s in chunk:
                pad = T - MEM - len(s)
                e = self.embed(torch.tensor(s, device=self.device))
                parts = [e, memw]
                if pad:
                    parts.insert(0, torch.zeros(pad, HIDDEN, dtype=e.dtype, device=self.device))
                embs.append(torch.cat(parts, 0))
                masks.append([0] * pad + [1] * (T - pad))
            x = torch.stack(embs)
            mask = torch.tensor(masks, device=self.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            h = self.inner(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                           use_cache=False).last_hidden_state
            outs.extend(h[:, -MEM:, :].unbind(0))
        res = [[] for _ in histories]
        for o, h in zip(outs, owner):
            res[h].append(o)
        return [torch.cat(r, 0) if r else torch.zeros(0, HIDDEN, dtype=torch.bfloat16,
                                                       device=self.device) for r in res]

    # ---------------------------------------------------------------- decoder
    def assemble(self, parts):
        """parts: list of ("ids", list[int]) | ("emb", tensor[n, 4096]). -> [N, 4096] bf16."""
        xs = []
        for kind, v in parts:
            if kind == "ids":
                if len(v):
                    xs.append(self.embed(torch.tensor(v, device=self.device)))
            else:
                if v.shape[0]:
                    xs.append(v.to(torch.bfloat16))
        return torch.cat(xs, 0)

    @torch.no_grad()
    def generate(self, inputs, greedy=True, temperature=0.7, top_p=0.95, max_new_tokens=512,
                 generator=None, prefill_chunk=4096):
        """Batched decoding with the adapter disabled.
        inputs: list of [N_i, 4096] decoder input embeddings. Each prompt is prefilled on its
        own (no padding compute), then the KV caches are left-padded into one batch for decoding.
        Returns list of generated token-id lists (EOS not included)."""
        B = len(inputs)
        lens = [x.shape[0] for x in inputs]
        Lmax = max(lens)
        cfg = self.inner.config
        nkv, hd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        K = [torch.zeros(B, nkv, Lmax, hd, dtype=torch.bfloat16, device=self.device)
             for _ in range(cfg.num_hidden_layers)]
        V = [torch.zeros_like(k) for k in K]
        first_logits = []
        with self.peft.disable_adapter():
            for i, x in enumerate(inputs):
                cache = DynamicCache()
                for c0 in range(0, lens[i], prefill_chunk):   # chunked prefill: lower peak memory
                    c1 = min(c0 + prefill_chunk, lens[i])
                    h = self.inner(inputs_embeds=x[None, c0:c1], use_cache=True,
                                   past_key_values=cache,
                                   position_ids=torch.arange(c0, c1, device=self.device)[None]
                                   ).last_hidden_state
                first_logits.append(self.lm_head(h[:, -1]).float())
                for l, layer in enumerate(cache.layers):
                    K[l][i, :, Lmax - lens[i]:] = layer.keys[0]
                    V[l][i, :, Lmax - lens[i]:] = layer.values[0]
                del cache, h
            def handover():   # DynamicLayer.update copies (torch.cat); free each padded layer
                while K:          # right after its copy so peak memory grows by one layer only
                    yield K.pop(0), V.pop(0)
            cache = DynamicCache(ddp_cache_data=handover())
            mask = torch.zeros(B, Lmax, dtype=torch.long, device=self.device)
            for i in range(B):
                mask[i, Lmax - lens[i]:] = 1
            pos = torch.tensor(lens, device=self.device)
            logits = torch.cat(first_logits, 0)
            out = [[] for _ in range(B)]
            alive = list(range(B))     # original indices of rows still in the batch
            for t in range(max_new_tokens):
                nxt = self._pick(logits, greedy, temperature, top_p, generator)
                keep = []
                for r, i in enumerate(alive):
                    tid = int(nxt[r])
                    if tid == EOS:
                        continue
                    out[i].append(tid)
                    keep.append(r)
                if not keep or t == max_new_tokens - 1:
                    break
                if len(keep) < len(alive):
                    idx = torch.tensor(keep, device=self.device)
                    cache.batch_select_indices(idx)
                    mask, pos, nxt = mask[idx], pos[idx], nxt[idx]
                    alive = [alive[r] for r in keep]
                mask = torch.cat([mask, torch.ones(len(alive), 1, dtype=mask.dtype,
                                                   device=self.device)], 1)
                h = self.inner(inputs_embeds=self.embed(nxt)[:, None], attention_mask=mask,
                               position_ids=pos[:, None], past_key_values=cache,
                               use_cache=True).last_hidden_state
                pos = pos + 1
                logits = self.lm_head(h[:, -1]).float()
        return out

    @staticmethod
    def _pick(logits, greedy, temperature, top_p, generator):
        if greedy:
            return logits.argmax(-1)
        probs = torch.softmax(logits / temperature, -1)
        sp, si = probs.sort(-1, descending=True)
        cum = sp.cumsum(-1)
        sp = sp.masked_fill(cum - sp > top_p, 0.0)   # keep the smallest set with mass >= top_p
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=generator)
        return si.gather(-1, choice).squeeze(-1)
