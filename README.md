# Llama2 Inference Runtime in C++ (RoPE, GQA, SwiGLU, KV Cache)

A from-scratch C++ inference engine for Llama 2 transformer models. Implements the full autoregressive forward pass tokenization, rotary positional embeddings, grouped-query attention, SwiGLU feed-forward networks, and BPE encoding without any deep learning framework dependency. Model configuration and vocabulary are serialized using Protocol Buffers.

---

## Architecture Overview
<img width="1164" height="4394" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/26efaa32-8d28-4d5c-94cb-a7ea7f8bd694" />


### Transformer Forward Pass

Each token processed by `forward()` runs through the following stages:

**1. Token Embedding Lookup**

The input token index is used to select a row from the embedding matrix `token_embedding [vocab_size x dim]`. This row is copied directly into the residual stream vector `x`.

**2. RMS Normalization**

Before both the attention and feed-forward sublayers, the residual stream is normalized using Root Mean Square Layer Normalization:

```
RMSNorm(x)_i = x_i / sqrt( mean(x^2) + eps ) * w_i
```

Unlike LayerNorm, RMSNorm omits the mean-centering step. The epsilon (`1e-6`) prevents division by zero for near-zero inputs. Each layer has separate learned scale vectors `rms_attention` and `rms_ffn`.

**3. QKV Projections**

The normalized residual is linearly projected into query, key, and value vectors:

- `Q = xb @ wQ`  — shape `[dim]`, one vector per attention head
- `K = xb @ wK`  — shape `[kv_dim]`, one vector per KV head
- `V = xb @ wV`  — shape `[kv_dim]`, one vector per KV head

`kv_dim = n_kv_heads * head_size`, which is smaller than `dim` when grouped-query attention is used.

**4. Rotary Positional Embedding (RoPE)**

Rather than adding positional encodings to the residual stream, Llama 2 rotates Q and K in-place using position-dependent rotation matrices. For each pair of adjacent dimensions `(i, i+1)` within a head:

```
freq  = 1 / 10000^( (i mod head_size) / head_size )
angle = position * freq
q[i]   = q[i]   * cos(angle) - q[i+1] * sin(angle)
q[i+1] = q[i]   * sin(angle) + q[i+1] * cos(angle)
```

The same rotation is applied to K. Using `i mod head_size` (rather than `i` directly) ensures the frequency resets at each head boundary, which is the convention in the reference implementation.

**5. KV Cache**

After RoPE, K and V are written into flat arrays `key_cache` and `value_cache` at offset `layer * seq_len * kv_dim + position * kv_dim`. This avoids recomputing projections for all past tokens on each forward step — the defining property of efficient autoregressive inference.

**6. Grouped-Query Attention (GQA)**

Standard multi-head attention uses one KV head per Q head. GQA allows fewer KV heads, where each KV head is shared by `kv_mul = num_heads / n_kv_heads` query heads. For head `h`, the KV head index is `h / kv_mul`.

For each query head, attention scores over all past positions `t` are:

```
score(t) = dot(q_h, k_h_t) / sqrt(head_size)
```

Scores are passed through softmax to produce attention weights, then used to compute a weighted sum of value vectors. The result is written into `xb`.

**7. Output Projection and Residual**

The concatenated attention output `xb` is projected back to `[dim]` via `wO`, then added to the residual stream:

```
x = x + wO @ xb
```

**8. SwiGLU Feed-Forward Network**

Each layer applies a gated feed-forward network with hidden dimension `hidden_dim` (typically `~2.7 * dim`). After RMS normalization:

```
gate   = w1 @ xb          # shape [hidden_dim]
up     = w3 @ xb          # shape [hidden_dim]
hidden = SiLU(gate) * up  # elementwise gated activation
output = w2 @ hidden      # shape [dim]
x      = x + output
```

`SiLU(x) = x * sigmoid(x)`. The gating mechanism (element-wise product of a SiLU-activated branch and a linear branch) is what distinguishes SwiGLU from a plain FFN. It improves training stability and final model quality at the same parameter count.

**9. Final Norm and Classifier**

After all layers, the residual stream undergoes one final RMSNorm, then is projected to `[vocab_size]` logits via `wcls`. When `shared_weights` is set in the checkpoint header, `wcls` is aliased to the token embedding matrix (weight tying).

---

### BPE Tokenizer

The tokenizer implements Byte Pair Encoding with a greedy merge strategy:

1. The input string is seeded character by character. Unknown characters fall back to `(unsigned char)c + 3` as a byte-level index.
2. All adjacent token pairs are scored by looking up their concatenation in the vocabulary. The highest-scoring pair is merged.
3. This repeats until no more merges exist in the vocabulary.
4. A BOS token (index 1) is always prepended.

Decoding handles two special cases: stripping a leading space when the previous token was BOS (index 1), and expanding byte-literal pieces of the form `<0xHH>` to their actual byte value.

---

### Weight Layout

Weights are read from a flat binary checkpoint. After a 7-integer header `(dim, hidden_dim, num_layers, num_heads, n_kv_heads, vocab_size, seq_len)`, weights appear in this order:

```
token_embedding   [vocab_size x dim]
rms_attention     [num_layers x dim]
wQ                [num_layers x dim x dim]
wK                [num_layers x dim x kv_dim]
wV                [num_layers x dim x kv_dim]
wO                [num_layers x dim x dim]
rms_ffn           [num_layers x dim]
w1                [num_layers x dim x hidden_dim]
w2                [num_layers x hidden_dim x dim]
w3                [num_layers x dim x hidden_dim]
rms_final         [dim]
freq_cis_real     [seq_len x head_size]   -- skipped
freq_cis_imag     [seq_len x head_size]   -- skipped
wcls              [vocab_size x dim]       -- only if not shared
```

The two `freq_cis` blocks are skipped because RoPE is computed on the fly. A negative `vocab_size` in the header signals weight sharing between the embedding table and the classifier.

---


## Building

### Prerequisites

Install dependencies on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y cmake protobuf-compiler libprotobuf-dev libgtest-dev
```

### Compile

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces the `run_main` binary inside `build/`.

---

## Running Inference

```bash
./build/run_main <model_checkpoint> <tokenizer_bin> "<prompt>"
```

Example with a 15M parameter Llama 2 model:

```bash
./build/run_main model.bin tokenizer.bin "Once upon a time"
```
---

## Running Tests

Tests cover `matmul`, `rmsnorm`, `softmax`, `encode`, and `decode` using GoogleTest. Build and run with:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./run_tests
```

---

## File Structure

```
.
├── CMakeLists.txt
├── llama_model.proto        # Protobuf schema for Config and TokenizerData
├── llama_infer.h            # Public API: structs, forward declarations
├── llama_infer.cc           # Core implementation
├── llama_infer_main.cc      # Entry point: load model, run generation loop
└── llama_infer_test.cpp     # Unit tests
```

---

## Numerical Notes

- All computation is in 32-bit float. No quantization is applied.
- Softmax uses the standard max-subtraction trick to prevent `exp` overflow.
- RMSNorm uses `eps = 1e-6` inside the square root.
- The KV cache is pre-allocated for the full `seq_len` at startup. Memory usage scales as `2 * num_layers * seq_len * kv_dim * 4` bytes.
- Matrix multiplication is naive (`O(n*d)` inner loops, no BLAS). For large models, this is the dominant cost.

---

## References

- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- Dauphin et al., "Language Modeling with Gated Convolutional Networks" — SwiGLU lineage
- Andrej Karpathy, `llama2.c` — reference C implementation and checkpoint format
