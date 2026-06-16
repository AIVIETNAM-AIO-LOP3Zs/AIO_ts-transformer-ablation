# Transformer Modules — Correctness Review

**Scope:** `src/ts_ablation/models/` (attention, feed-forward, embedding, encoder, decoder) plus `Dataloader.py`.
**Status:** All existing tests in `test.py` pass. Findings below are from static review of the wiring confirmed via GitNexus (`DataEmbedding → Encoder(EncoderLayer×N) → Decoder(DecoderLayer×N)`, each layer using `MultiHeadAttention` + `FeedForward`).

---

## Severity summary

> **Status: all 7 items resolved.** Verified with `uv run` across every module `main()`, `test.py`, and `Dataloader.py`. `detect_changes` reports LOW risk, 0 affected processes.

| # | Severity | Module | Problem | Status |
|---|----------|--------|---------|--------|
| 1 | 🔴 Bug | `encoder_layer.py` | `attn_mask` silently dropped when `output_attention=True` | ✅ Fixed |
| 2 | 🔴 Bug | `decoder.py` / `multi_head_attention.py` | SDPA receives both `is_causal=True` **and** `attn_mask` → crashes when a decoder mask is passed | ✅ Fixed |
| 3 | 🟡 Design | `multi_head_attention.py` + layers | Double dropout on every attention sub-layer | ✅ Fixed |
| 4 | 🟡 Design | `multi_head_attention.py` | `output_attention=True` recomputes attention from scratch (distorts efficiency metrics) | ✅ Fixed |
| 5 | 🟡 Latent | `embedding.py` | Fragile lexicographic torch-version check for Conv1d padding | ✅ Fixed |
| 6 | 🟢 Minor | `embedding.py` | `create_sin_cos_matrix` breaks for odd `d_model` | ✅ Fixed |
| 7 | 🟢 Minor | `Dataloader.py` | Dataset emits no temporal marks; pipeline not fully wired | ✅ Fixed |

---

## ✅ What is correct

- **Pre-LN** applied consistently across `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, including the required final `LayerNorm` after each stack.
- **Causal masking verified** — `test.py` leakage check passes (max past-difference = `0.0` when future inputs are perturbed).
- **Shapes correct** — attention `(B, H, T_q, T_k)`, cross-attention `(B, H, L_dec, L_enc)`, outputs `(B, T, d_model)`.
- **Cross-attention** correctly uses `is_causal=False` with Q from decoder, K/V from encoder memory.
- `split_heads` / `combine_heads` are correct inverse operations; `FeedForward` is textbook-correct.

---

## 🔴 Bug 1 — `EncoderLayer` drops `attn_mask` when `output_attention=True`

**File:** `src/ts_ablation/models/encoder_layer.py:67-71`

```python
if self.output_attention:
    x, attn_weights = self.attention(x)          # ← attn_mask NOT passed
else:
    x = self.attention(x, attn_mask=attn_mask)   # ← mask only applied here
```

`Encoder.forward` passes `attn_mask` to every layer, but the moment attention visualization is enabled the mask is **silently ignored** — masked and unmasked runs diverge with no error.

**Fix:** pass the mask in both branches.

```python
if self.output_attention:
    x, attn_weights = self.attention(x, attn_mask=attn_mask)
else:
    x = self.attention(x, attn_mask=attn_mask)
    attn_weights = None
```

---

## 🔴 Bug 2 — Decoder self-attention passes both `is_causal` and `attn_mask` to SDPA

**Files:** `src/ts_ablation/models/decoder.py:96-103`, `src/ts_ablation/models/multi_head_attention.py:68-73`

```python
self.self_attention(queries=x, keys=x, values=x, attn_mask=x_mask, is_causal=True)
```

`torch.nn.functional.scaled_dot_product_attention` **raises** when `attn_mask is not None` *and* `is_causal=True`. It works today only because `x_mask` defaults to `None`; supplying a decoder padding mask will crash at runtime.

**Fix:** merge the causal mask and padding mask into a single additive/boolean `attn_mask`, then call SDPA with `is_causal=False`.

```python
# build a combined mask once, then:
self.self_attention(queries=x, keys=x, values=x,
                    attn_mask=combined_mask, is_causal=False)
```

---

## 🟡 Design 3 — Double dropout on every attention sub-layer

**Files:** `multi_head_attention.py:78`, `encoder_layer.py:73`, `decoder.py:108,136`

`MultiHeadAttention.forward` applies `self.dropout(out)` after `fc_out`, then the enclosing layer applies `dropout1/2/3` again before the residual add. Two stacked dropouts give an effective rate of `1 - (1 - p)²` (≈ 0.19 at `p=0.1`). Harmless to shapes, but it contaminates any dropout/regularization ablation.

**Fix:** keep dropout in exactly one place (conventionally inside the sub-layer, removing the layer-level `dropoutN`, or vice-versa).

---

## 🟡 Design 4 — `output_attention=True` recomputes attention from scratch

**File:** `src/ts_ablation/models/multi_head_attention.py:85-98`

The output is produced by fused `F.scaled_dot_product_attention`, but the returned weights are a **second** independent `matmul + softmax`. Consequences:

- Reported weights don't reflect the dropout actually applied to the output.
- Attention FLOPs are doubled whenever weights are requested — directly distorting the **computational-efficiency** metric this ablation study measures.

**Fix:** when weights are needed, compute attention manually once (scores → softmax → dropout → `@V`) and derive both the output and the weights from that single pass.

---

## 🟡 Latent 5 — Fragile torch-version check for Conv1d padding

**File:** `src/ts_ablation/models/embedding.py:40`

```python
padding = 1 if torch.__version__ >= '1.5.0' else 2
```

This is a **lexicographic string comparison**: `'1.10.0' < '1.5.0'` evaluates to `True`, so torch 1.10–1.13 would select `padding=2`, making the Conv1d output length `L+2` and breaking the element-wise add in `DataEmbedding`. Currently resolves to `1` on torch 2.x, but it is a latent trap.

**Fix:** hardcode `padding = 1` for `kernel_size=3` (preserves sequence length).

---

## 🟢 Minor 6 — `create_sin_cos_matrix` breaks for odd `d_model`

**File:** `src/ts_ablation/models/embedding.py:18-20`

For odd `d_model`, `weight[:, 1::2]` has fewer columns than `div_term`, causing a broadcast/shape error. Fine at `d_model=512`, but an ablation sweeping the model dimension could hit it.

**Fix:** slice `div_term` to match the cosine slice, e.g. `weight[:, 1::2] = torch.cos(position * div_term[: weight[:, 1::2].size(1)])`.

---

## 🟢 Minor 7 — Pipeline not fully wired

**File:** `Dataloader.py:46-72`

`DataEmbedding.forward(a, a_mark)` requires temporal marks, but `ETTDataset` emits only `x_enc`, `x_dec`, `y` — no `x_mark`. There is also no top-level module joining embedding → encoder → decoder → projection head yet (GitNexus reports 0 execution flows). Expected if assembly is still pending, but the temporal-mark mismatch must be addressed when the full model is built.

---

## Recommended fix order

1. **Bug 2** (decoder SDPA conflict) — blocks any masked decoder experiment.
2. **Bug 1** (encoder mask drop) — blocks masked + attention-visualization runs.
3. **Design 3 & 4** — required for clean dropout / efficiency ablation metrics.
4. **Latent 5, Minor 6, 7** — hardening before scaling the experiment grid.
