#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "llama_infer.h"
#include <omp.h>

namespace llama_infer {

// allocState — mirrors allocState() in the reference C code exactly.
// kv_dim = (dim * n_kv_heads) / num_heads
static void allocState(LlamaRunState* s, const Config& c)
{
    int kv_dim = (c.dim() * c.n_kv_heads()) / c.num_heads();

    s->x.resize(c.dim());
    s->xb.resize(c.dim());
    s->xb2.resize(c.dim());

    s->q.resize(c.dim());
    s->k.resize(kv_dim);
    s->v.resize(kv_dim);

    s->hb.resize(c.hidden_dim());
    s->hb2.resize(c.hidden_dim());

    s->attention.resize(c.num_heads() * c.seq_len());
    s->logits.resize(c.vocab_size());

    s->key_cache.resize((size_t)c.num_layers() * c.seq_len() * kv_dim);
    s->value_cache.resize((size_t)c.num_layers() * c.seq_len() * kv_dim);
}

// wireWeights — pointer layout must match llama2.c export.py exactly.
//
// Checkpoint binary layout (floats, after the 7-int header):
//
//   token_embedding  [vocab_size * dim]
//   rms_attention    [num_layers * dim]
//   wQ               [num_layers * dim * dim]            (dim == num_heads * head_size)
//   wK               [num_layers * dim * kv_dim]
//   wV               [num_layers * dim * kv_dim]
//   wO               [num_layers * dim * dim]
//   rms_ffn          [num_layers * dim]
//   w1               [num_layers * dim * hidden_dim]
//   w2               [num_layers * hidden_dim * dim]
//   w3               [num_layers * dim * hidden_dim]
//   rms_final        [dim]
//   <skip>           [seq_len * head_size]               -- freq_cis real
//   <skip>           [seq_len * head_size]               -- freq_cis imag
//   wcls             [vocab_size * dim]                  (only when !shared)
//
// The freq_cis skip is present in older checkpoints. We always skip it to
// match the reference C implementation; if the checkpoint was built without
// it the skip pushes wcls into garbage — but those checkpoints set
// vocab_size < 0 (shared), so wcls is never read from the file.
static void wireWeights(LlamaWeights* w, const Config& c,
                        float* ptr, int shared)
{
    const int dim       = c.dim();
    const int hidden    = c.hidden_dim();
    const int layers    = c.num_layers();
    const int n_heads   = c.num_heads();
    const int n_kv      = c.n_kv_heads();
    const int vocab     = c.vocab_size();
    const int head_size = dim / n_heads;          // per-head dimension
    const int kv_dim    = n_kv * head_size;       // key/value width

    w->token_embedding = ptr;  ptr += (long)vocab  * dim;
    w->rms_attention   = ptr;  ptr += (long)layers * dim;

    // wQ: each layer is [dim, dim]
    w->wQ = ptr;  ptr += (long)layers * dim * dim;

    // wK / wV: each layer is [dim, kv_dim]
    w->wK = ptr;  ptr += (long)layers * dim * kv_dim;
    w->wV = ptr;  ptr += (long)layers * dim * kv_dim;

    // wO: each layer is [dim, dim]  (same shape as wQ in standard llama2)
    w->wO = ptr;  ptr += (long)layers * dim * dim;

    w->rms_ffn = ptr;  ptr += (long)layers * dim;

    // FFN weights
    w->w1 = ptr;  ptr += (long)layers * dim    * hidden;
    w->w2 = ptr;  ptr += (long)layers * hidden * dim;
    w->w3 = ptr;  ptr += (long)layers * dim    * hidden;

    w->rms_final = ptr;  ptr += dim;

    // Skip the precomputed RoPE freq_cis tables that older export scripts
    // embed in the file.  The reference C code always skips these two blocks:
    //   ptr += seq_len * head_size;   // freq_cis real
    //   ptr += seq_len * head_size;   // freq_cis imag
    ptr += (long)c.seq_len() * head_size;
    ptr += (long)c.seq_len() * head_size;

    // When shared (positive vocab_size in the raw header), the classifier
    // weight matrix reuses the token embedding table.
    w->wcls = shared ? w->token_embedding : ptr;

    fprintf(stderr,
            "wireWeights: token_emb=%p  rms_final=%p  wcls=%p  shared=%d\n",
            (void*)w->token_embedding, (void*)w->rms_final,
            (void*)w->wcls, shared);
}

// On-disk header layout (7 × int32).
struct BinaryConfig {
    int32_t dim, hidden_dim, num_layers, num_heads, n_kv_heads, vocab_size, seq_len;
};

// Function to load the transformer model from a binary file, including the header and weights.
void loadTransformer(LlamaTransformer& t, const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open '%s'\n", path); exit(1); }

    BinaryConfig bc{};
    if (fread(&bc, sizeof(bc), 1, f) != 1) {
        fprintf(stderr, "ERROR: failed to read model header\n"); exit(1);
    }

    fprintf(stderr,
            "Header: dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
            bc.dim, bc.hidden_dim, bc.num_layers, bc.num_heads,
            bc.n_kv_heads, bc.vocab_size, bc.seq_len);

    int shared = (bc.vocab_size > 0) ? 1 : 0;
    int vocab  = std::abs(bc.vocab_size);

    t.config.set_dim(bc.dim);
    t.config.set_hidden_dim(bc.hidden_dim);
    t.config.set_num_layers(bc.num_layers);
    t.config.set_num_heads(bc.num_heads);
    t.config.set_n_kv_heads(bc.n_kv_heads);
    t.config.set_vocab_size(vocab);
    t.config.set_seq_len(bc.seq_len);

    // Read the entire weight blob into a flat float buffer.
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long wbytes    = file_size - (long)sizeof(BinaryConfig);
    if (wbytes <= 0) {
        fprintf(stderr, "ERROR: no weight data after header\n"); exit(1);
    }
    fseek(f, sizeof(BinaryConfig), SEEK_SET);

    // Sanity-check: compute expected byte count and warn if mismatched.
    {
        const int hs     = bc.dim / bc.num_heads;
        const int kv_dim = bc.n_kv_heads * hs;
        long exp = 0;
        exp += (long)vocab             * bc.dim;                          // token_emb
        exp += (long)bc.num_layers     * bc.dim;                          // rms_att
        exp += (long)bc.num_layers     * bc.dim  * bc.dim;                // wQ
        exp += (long)bc.num_layers     * bc.dim  * kv_dim;                // wK
        exp += (long)bc.num_layers     * bc.dim  * kv_dim;                // wV
        exp += (long)bc.num_layers     * bc.dim  * bc.dim;                // wO
        exp += (long)bc.num_layers     * bc.dim;                          // rms_ffn
        exp += (long)bc.num_layers     * bc.dim         * bc.hidden_dim;  // w1
        exp += (long)bc.num_layers     * bc.hidden_dim  * bc.dim;         // w2
        exp += (long)bc.num_layers     * bc.dim         * bc.hidden_dim;  // w3
        exp += (long)bc.dim;                                              // rms_final
        exp += (long)bc.seq_len * hs;                                     // freq_cis re
        exp += (long)bc.seq_len * hs;                                     // freq_cis im
        if (!shared) exp += (long)vocab * bc.dim;                         // wcls

        fprintf(stderr,
                "Weight bytes in file: %ld | Expected: %ld floats = %ld bytes | %s\n",
                wbytes, exp, exp * 4,
                (wbytes == exp * 4) ? "OK" : "MISMATCH — check checkpoint format");
    }

    size_t n_floats = (size_t)wbytes / sizeof(float);
    t.data.buffer = std::make_unique<float[]>(n_floats);
    t.data.size   = (size_t)wbytes;

    if (fread(t.data.buffer.get(), 1, (size_t)wbytes, f) != (size_t)wbytes) {
        fprintf(stderr, "ERROR: short read on weight data\n"); exit(1);
    }
    fclose(f);

    wireWeights(&t.weights, t.config, t.data.buffer.get(), shared);
    allocState(&t.state, t.config);
}

// Function to load the tokenizer data from a binary file, including the vocabulary and scores.
void loadTokenizer(TokenizerData* t, const char* path, int vocab_size)
{
    t->set_vocab_sized(vocab_size);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open tokenizer '%s'\n", path); exit(1); }

    int32_t max_token_len = 0;
    fread(&max_token_len, sizeof(int32_t), 1, f);
    t->set_max_token_len(max_token_len);

    for (int i = 0; i < vocab_size; i++) {
        float   score = 0.0f;
        int32_t len   = 0;
        fread(&score, sizeof(float),   1, f);
        fread(&len,   sizeof(int32_t), 1, f);
        if (len <= 0 || len > 512) {
            fprintf(stderr, "ERROR: bad token length %d at index %d\n", len, i);
            exit(1);
        }
        std::string tok(len, '\0');
        fread(&tok[0], 1, (size_t)len, f);
        t->add_scores(score);
        t->add_vocab(tok);
    }
    fclose(f);
    fprintf(stderr, "Tokenizer: %d tokens loaded\n", vocab_size);
}

// Function to look up a token string in the tokenizer's vocabulary and return its index, or -1 if not found.
static int vocabLookup(const TokenizerData* t, const std::string& s)
{
    for (int i = 0; i < t->vocab_sized(); i++)
        if (t->vocab(i) == s) return i;
    return -1;
}

// Function to encode a text string into a sequence of token indices using the tokenizer's vocabulary and BPE merges.
int encode(const TokenizerData* t, const char* text, int* tokens)
{
    int n = 0;

    // BOS token is always index 1.
    tokens[n++] = 1;

    // Seed with single-character tokens (or raw byte fallback).
    for (const char* c = text; *c; c++) {
        std::string buf(1, *c);
        int id = vocabLookup(t, buf);
        if (id != -1) {
            tokens[n++] = id;
        } else {
            // Match C: (unsigned char)*c + 3, clamped to valid range.
            int fb = static_cast<unsigned char>(*c) + 3;
            tokens[n++] = (fb < t->vocab_sized()) ? fb : (t->vocab_sized() - 1);
        }
    }

    // Greedy BPE merges: repeatedly find the highest-score adjacent pair.
    while (true) {
        float best    = -1e10f;
        int   best_id = -1, best_pos = -1;

        for (int i = 0; i < n - 1; i++) {
            if (tokens[i]   < 0 || tokens[i]   >= t->vocab_sized()) continue;
            if (tokens[i+1] < 0 || tokens[i+1] >= t->vocab_sized()) continue;
            std::string merged = t->vocab(tokens[i]) + t->vocab(tokens[i+1]);
            int id = vocabLookup(t, merged);
            if (id != -1 && t->scores(id) > best) {
                best     = t->scores(id);
                best_id  = id;
                best_pos = i;
            }
        }
        if (best_pos == -1) break;   // no more merges

        tokens[best_pos] = best_id;
        // Shift remaining tokens left.
        for (int i = best_pos + 1; i < n - 1; i++) tokens[i] = tokens[i+1];
        n--;
    }
    return n;
}

// Function to decode a single token index back into a string, handling special cases for leading spaces and byte-encoded pieces.
std::string decode(const TokenizerData* t, int prev, int cur)
{
    if (cur < 0 || cur >= t->vocab_sized()) return "";

    std::string s = t->vocab(cur);

    // Strip leading space when previous token was BOS (id 1).
    if (prev == 1 && !s.empty() && s[0] == ' ') s = s.substr(1);

    // Expand byte-encoded pieces like "<0x0A>" → the actual byte.
    unsigned char byte_val = 0;
    if (s.size() == 6 && sscanf(s.c_str(), "<0x%02hhX>", &byte_val) == 1)
        return std::string(1, static_cast<char>(byte_val));

    return s;
}

void matmul(std::vector<float>& out,
            const std::vector<float>& x,
            const float* W, int n, int d)
{
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += x[j] * W[i * n + j];
        out[i] = val;
    }
}

// Function to perform RMS normalization on the input vector x using the provided weights w, and store the result in out.
void rmsnorm(std::vector<float>& out,
             const std::vector<float>& x,
             const float* w, int n)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
}

// Function to apply the softmax function to the input vector x of length n, modifying x in place to contain the resulting probabilities.
void softmax(std::vector<float>& x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// In-place version of softmax that operates directly on a float pointer, modifying the values in place.
static void softmax_inplace(float* x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Function to perform the forward pass of the transformer model for a single token at a given position, updating the state and returning the output logits.
std::vector<float>& forward(LlamaTransformer& transformer, int token, int pos)
{
    const Config&  config  = transformer.config;
    LlamaWeights&  weights = transformer.weights;
    LlamaRunState& state   = transformer.state;

    const int dim       = config.dim();
    const int kv_dim    = (config.n_kv_heads() * dim) / config.num_heads();
    const int head_size = dim / config.num_heads();
    const int kv_mul    = config.num_heads() / config.n_kv_heads();
    const int hidden    = config.hidden_dim();

    // Copy token embedding into residual stream.
    const float* emb = weights.token_embedding + (long)token * dim;
    std::copy(emb, emb + dim, state.x.begin());

    for (int l = 0; l < config.num_layers(); l++) {

        // Attention pre-norm
        rmsnorm(state.xb, state.x, weights.rms_attention + l * dim, dim);

        // Q projection: [dim] → [dim]
        matmul(state.q, state.xb, weights.wQ + (long)l * dim * dim, dim, dim);

        // RoPE on Q: loop over dim, use i % head_size for the frequency index.
        // This matches: for (int i = 0; i < dim; i += 2) in the C code.
        {
            float* q = state.q.data();
            for (int i = 0; i < dim; i += 2) {
                float freq  = 1.0f / powf(10000.0f,
                                          (float)(i % head_size) / (float)head_size);
                float angle = (float)pos * freq;
                float c_    = cosf(angle);
                float s_    = sinf(angle);
                float v0    = q[i], v1 = q[i+1];
                q[i]   = v0 * c_ - v1 * s_;
                q[i+1] = v0 * s_ + v1 * c_;
            }
        }

        // K projection: [dim] → [kv_dim]
        matmul(state.k, state.xb, weights.wK + (long)l * dim * kv_dim, dim, kv_dim);

        // V projection: [dim] → [kv_dim]
        matmul(state.v, state.xb, weights.wV + (long)l * dim * kv_dim, dim, kv_dim);

        // RoPE on K: loop over kv_dim, use i % head_size.
        {
            float* k = state.k.data();
            for (int i = 0; i < kv_dim; i += 2) {
                float freq  = 1.0f / powf(10000.0f,
                                          (float)(i % head_size) / (float)head_size);
                float angle = (float)pos * freq;
                float c_    = cosf(angle);
                float s_    = sinf(angle);
                float v0    = k[i], v1 = k[i+1];
                k[i]   = v0 * c_ - v1 * s_;
                k[i+1] = v0 * s_ + v1 * c_;
            }
        }

        // Store K and V into the KV cache for this layer and position.
        const long loff  = (long)l * config.seq_len() * kv_dim;
        float* kc = state.key_cache.data()   + loff + (long)pos * kv_dim;
        float* vc = state.value_cache.data()  + loff + (long)pos * kv_dim;
        std::copy(state.k.begin(), state.k.end(), kc);
        std::copy(state.v.begin(), state.v.end(), vc);

        // Multi-head attention
        #pragma omp parallel for
        for (int h = 0; h < config.num_heads(); h++) {
            float* q_h   = state.q.data()         + h * head_size;
            float* att_h = state.attention.data()  + h * config.seq_len();
            float* xb_h  = state.xb.data()         + h * head_size;

            // Compute attention scores for all past positions.
            for (int t = 0; t <= pos; t++) {
                // GQA: head h maps to KV head h / kv_mul.
                float* k_h = state.key_cache.data()
                             + loff + (long)t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q_h[i] * k_h[i];
                att_h[t] = score / sqrtf((float)head_size);
            }

            // Softmax over [0..pos].
            softmax_inplace(att_h, pos + 1);

            // Weighted sum of value vectors.
            std::fill(xb_h, xb_h + head_size, 0.0f);
            for (int t = 0; t <= pos; t++) {
                float* v_h = state.value_cache.data()
                             + loff + (long)t * kv_dim + (h / kv_mul) * head_size;
                float a = att_h[t];
                for (int i = 0; i < head_size; i++) xb_h[i] += a * v_h[i];
            }
        }

        // wO projection: [dim] → [dim]  (stride = dim * dim per layer)
        matmul(state.xb2, state.xb, weights.wO + (long)l * dim * dim, dim, dim);

        // Residual connection.
        for (int i = 0; i < dim; i++) state.x[i] += state.xb2[i];

        // FFN (SwiGLU)
        rmsnorm(state.xb, state.x, weights.rms_ffn + l * dim, dim);

        // Gate  (w1) and up-projection (w3).
        matmul(state.hb,  state.xb, weights.w1 + (long)l * dim * hidden, dim, hidden);
        matmul(state.hb2, state.xb, weights.w3 + (long)l * dim * hidden, dim, hidden);

        // SiLU gate: hb[i] = silu(hb[i]) * hb2[i]
        for (int i = 0; i < hidden; i++) {
            float v     = state.hb[i];
            state.hb[i] = (v / (1.0f + expf(-v))) * state.hb2[i];
        }

        // Down-projection (w2): [hidden_dim] → [dim]
        matmul(state.xb, state.hb, weights.w2 + (long)l * hidden * dim, hidden, dim);

        // Residual connection.
        for (int i = 0; i < dim; i++) state.x[i] += state.xb[i];
    }

    // Final norm and classifier projection.
    rmsnorm(state.x, state.x, weights.rms_final, dim);
    matmul(state.logits, state.x, weights.wcls, dim, config.vocab_size());

    return state.logits;
}

} // namespace llama_infer