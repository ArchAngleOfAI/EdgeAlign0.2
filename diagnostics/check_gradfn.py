"""Test 2 (Hypothesis A): does the rollout keep an autograd graph at every step?

A COPY of distill_softprompt.generate_softprompt (same ops, same order) with
instrumentation: at steps 1, 50 and 100 it prints soft_token.grad_fn and
measures the autograd graph hanging off it -- number of unique grad_fn nodes,
longest node chain (depth), and which trainable leaves (AccumulateGrad nodes)
are reachable. If backprop-through-time is intact, all three should grow with
the step index, and step 100 should reach the last-layer weights via ~99
recurrent hops.
"""

import torch

from _common import NUM_SOFT_TOKENS, get_device, get_tokenizer, load_all, setup_stdout, take_batches

REPORT_STEPS = {0: "first (1)", NUM_SOFT_TOKENS // 2 - 1: f"middle ({NUM_SOFT_TOKENS // 2})",
                NUM_SOFT_TOKENS - 1: f"last ({NUM_SOFT_TOKENS})"}


def graph_stats(root, leaf_names):
    """Unique nodes, longest path (in nodes), and reachable named leaves under `root`."""
    nodes, order, stack = set(), [], [(root, False)]
    while stack:  # iterative post-order DFS
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if node is None or node in nodes:
            continue
        nodes.add(node)
        stack.append((node, True))
        for child, _ in node.next_functions:
            if child is not None and child not in nodes:
                stack.append((child, False))
    depth = {}
    for node in order:  # children precede parents in post-order
        depth[node] = 1 + max((depth.get(c, 0) for c, _ in node.next_functions if c is not None), default=0)
    leaves = {leaf_names.get(id(n.variable), "other")
              for n in nodes if type(n).__name__ == "AccumulateGrad"}
    return len(nodes), depth[root], sorted(leaves)


def generate_softprompt_instrumented(encoder_model, embedding2, prompt_ids, num_soft_tokens, leaf_names):
    inner = encoder_model.model
    embeds = inner.embed_tokens(prompt_ids)
    out = inner(inputs_embeds=embeds, use_cache=True)
    past_key_values = out.past_key_values
    hidden = out.last_hidden_state[:, -1:, :]

    soft_tokens = []
    for step in range(num_soft_tokens):
        logits = encoder_model.lm_head(hidden)
        probs = torch.softmax(logits, dim=-1)
        soft_token = probs @ embedding2.weight
        if step in REPORT_STEPS:
            n, d, leaves = graph_stats(soft_token.grad_fn, leaf_names)
            print(f"step {REPORT_STEPS[step]:>12s}: grad_fn={soft_token.grad_fn}  "
                  f"requires_grad={soft_token.requires_grad}")
            layer_leaves = [x for x in leaves if x.startswith("layers[")]
            other = [x for x in leaves if not x.startswith("layers[")]
            print(f"{'':16s}graph nodes={n:,}  longest chain={d:,}  trainable leaves reached: "
                  f"{other} + {len(layer_leaves)} last-layer params")
        soft_tokens.append(soft_token)
        if step == num_soft_tokens - 1:
            break
        out = inner(inputs_embeds=soft_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        hidden = out.last_hidden_state
    return torch.cat(soft_tokens, dim=1)


def main():
    setup_stdout()
    device = get_device()
    encoder, embedding2, _, _, _ = load_all(device)
    prompt_ids = take_batches(get_tokenizer(), 1)[0].to(device)

    leaf_names = {id(embedding2.weight): "embedding2", id(encoder.lm_head.weight): "lm_head"}
    last_idx = len(encoder.model.layers) - 1
    for n, p in encoder.model.layers[-1].named_parameters():
        leaf_names[id(p)] = f"layers[{last_idx}].{n}"

    sp = generate_softprompt_instrumented(encoder, embedding2, prompt_ids, NUM_SOFT_TOKENS, leaf_names)
    print(f"\nsoftprompt grad_fn={sp.grad_fn}, shape={tuple(sp.shape)}")


if __name__ == "__main__":
    main()
