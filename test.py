import torch
import torch.nn.functional as F

# ── Test 1: LayerNorm ──────────────────────────────────────
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]],
                  [[9.0, 10.0, 11.0, 12.0],
                   [13.0, 14.0, 15.0, 16.0]]])  # [2, 2, 4]

weight = torch.ones(4)
bias   = torch.zeros(4)
out = F.layer_norm(x, [4], weight, bias)
print("=== LayerNorm Output ===")
print(out)

# ── Test 2: SelfAttention ──────────────────────────────────
x = torch.tensor([[[1.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0]]])  # [1, 3, 3] — batch=1, seq_len=3, d_in=3

W_q = torch.tensor([[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])  # [3, 2]

W_k = torch.tensor([[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0]])  # [3, 2]

W_v = torch.tensor([[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])  # [3, 2]

keys    = x @ W_k   # [1, 3, 2]
queries = x @ W_q   # [1, 3, 2]
values  = x @ W_v   # [1, 3, 2]

attn_scores  = queries @ keys.transpose(-2, -1)  # [1, 3, 3]
attn_weights = F.softmax(attn_scores * (1.0/2)**0.5, dim=-1)
context      = attn_weights @ values              # [1, 3, 2]

print("=== keys ===");    print(keys)
print("=== queries ==="); print(queries)
print("=== values ===");  print(values)
print("=== attn_scores ===");  print(attn_scores)
print("=== attn_weights ==="); print(attn_weights)
print("=== SelfAttention Output ==="); print(context)

# ── Test 3: CausalSelfAttention ───────────────────────────
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
causal_scores  = attn_scores.masked_fill(mask, float('-inf'))
causal_weights = F.softmax(causal_scores * (1.0/2)**0.5, dim=-1)
causal_context = causal_weights @ values

print("=== Causal attn_scores ===");  print(causal_scores)
print("=== Causal attn_weights ==="); print(causal_weights)
print("=== CausalSelfAttention Output ==="); print(causal_context)
