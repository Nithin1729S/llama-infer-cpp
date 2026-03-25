#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <memory>
#include <string>
#include "llama_infer.h"
#include "llama_model.pb.h"

namespace llama_infer {

struct Weights {
    float* token_embedding;
    float* rms_attention;
    float* wQ;
    float* wK;
    float* wV;
    float* wO;
    float* w1;
    float* w2;
    float* w3;
    float* rms_ffn;
    float* rms_final;
    float* wcls;
};

struct ModelData {
    std::unique_ptr<float[]> buffer;
    size_t size;
};

struct RunState {
    std::vector<float> x;
    std::vector<float> xb, xb2;
    std::vector<float> q, k, v;
    std::vector<float> hb, hb2;
    std::vector<float> attention;
    std::vector<float> logits;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

void allocState(RunState* state, const Config& config) {
    int kv_dimensions = (config.dim() * config.n_kv_heads()) / config.num_heads();

    state->x.resize(config.dim());
    state->xb.resize(config.dim());
    state->xb2.resize(config.dim());

    state->q.resize(config.dim());
    state->k.resize(kv_dimensions);
    state->v.resize(kv_dimensions);

    state->hb.resize(config.hidden_dim());
    state->hb2.resize(config.hidden_dim());

    state->attention.resize(config.num_heads() * config.seq_len());
    state->logits.resize(config.vocab_size());

    state->key_cache.resize(config.num_layers() * config.seq_len() * kv_dimensions);
    state->value_cache.resize(config.num_layers() * config.seq_len() * kv_dimensions);
}

// Owns all transformer state; populated by loadTransformer().
struct Transformer {
    Config    config;   // proto message
    Weights   weights;  // raw pointers into data.buffer
    RunState  state;    // working memory
    ModelData data;     // owns the weight buffer
};

// ---------------------------------------------------------------------------
// Math helpers (unchanged)
// ---------------------------------------------------------------------------

void matmul(std::vector<float>& out,
            const std::vector<float>& x,
            const float* W,
            int n, int d)
{
    for (int i = 0; i < d; i++) {
        float val = 0;
        for (int j = 0; j < n; j++)
            val += x[j] * W[i * n + j];
        out[i] = val;
    }
}

void rmsnorm(std::vector<float>& out,
             const std::vector<float>& x,
             const float* w,
             int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];

    ss = 1.0f / sqrtf(ss / n + 1e-6f);

    for (int i = 0; i < n; i++)
        out[i] = x[i] * ss * w[i];
}

void softmax(std::vector<float>& x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

// ---------------------------------------------------------------------------
// wireWeights (unchanged — works off raw float* after Config is populated)
// ---------------------------------------------------------------------------

void wireWeights(Weights* w, const Config& c, float* ptr, int shared)
{
    int dim      = c.dim();
    int hidden   = c.hidden_dim();
    int n_heads  = c.num_heads();
    int n_kv     = c.n_kv_heads();
    int seq_len  = c.seq_len();
    int vocab    = c.vocab_size();
    int layers   = c.num_layers();

    int head_size = dim / n_heads;
    int kv_dim    = (dim * n_kv) / n_heads;

    w->token_embedding = ptr; ptr += vocab  * dim;
    w->rms_attention   = ptr; ptr += layers * dim;
    w->wQ              = ptr; ptr += layers * dim * dim;
    w->wK              = ptr; ptr += layers * dim * kv_dim;
    w->wV              = ptr; ptr += layers * dim * kv_dim;
    w->wO              = ptr; ptr += layers * dim * dim;
    w->rms_ffn         = ptr; ptr += layers * dim;
    w->w1              = ptr; ptr += layers * dim * hidden;
    w->w2              = ptr; ptr += layers * hidden * dim;
    w->w3              = ptr; ptr += layers * dim * hidden;
    w->rms_final       = ptr; ptr += dim;

    ptr += seq_len * head_size; // freq_cis_real (skipped)
    ptr += seq_len * head_size; // freq_cis_imag (skipped)

    if (shared)
        w->wcls = w->token_embedding;
    else {
        w->wcls = ptr;
        ptr += vocab * dim;
    }
}

// ---------------------------------------------------------------------------
// loadTransformer
//
// The binary file stores a plain C struct at the start, NOT a serialised proto.
// We read it into BinaryConfig first, then populate the proto Config message.
// ---------------------------------------------------------------------------

// Mirrors the on-disk layout of the original llama2.c checkpoint header.
struct BinaryConfig {
    int32_t dim;
    int32_t hidden_dim;
    int32_t num_layers;
    int32_t num_heads;
    int32_t n_kv_heads;
    int32_t vocab_size; // negative value signals shared classifier weights
    int32_t seq_len;
};

void loadTransformer(Transformer& t, const char* path)
{
    FILE* file = fopen(path, "rb");
    if (!file) return;

    // 1. Read the raw binary header into a plain struct.
    BinaryConfig bc{};
    fread(&bc, sizeof(BinaryConfig), 1, file);

    // Negative vocab_size signals that the classifier matrix is shared with
    // the token-embedding table.
    int shared = (bc.vocab_size > 0) ? 1 : 0;

    // 2. Populate the proto Config message via its generated setters.
    t.config.set_dim(bc.dim);
    t.config.set_hidden_dim(bc.hidden_dim);
    t.config.set_num_layers(bc.num_layers);
    t.config.set_num_heads(bc.num_heads);
    t.config.set_n_kv_heads(bc.n_kv_heads);
    t.config.set_vocab_size(std::abs(bc.vocab_size));
    t.config.set_seq_len(bc.seq_len);

    // 3. Load the weight bytes.
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, sizeof(BinaryConfig), SEEK_SET);

    long wsize  = file_size - static_cast<long>(sizeof(BinaryConfig));
    t.data.size   = static_cast<size_t>(wsize);
    t.data.buffer = std::make_unique<float[]>(wsize / sizeof(float));

    fread(t.data.buffer.get(), 1, wsize, file);
    fclose(file);

    wireWeights(&t.weights, t.config, t.data.buffer.get(), shared);
    allocState(&t.state, t.config);
}

// ---------------------------------------------------------------------------
// loadTokenizer
//
// Populates a proto TokenizerData message instead of a raw Tokenizer struct.
// ---------------------------------------------------------------------------

static void loadTokenizer(TokenizerData* t, const char* path, int vocab_size)
{
    t->set_vocab_sized(vocab_size);   // note: field is named vocab_sized in proto

    FILE* f = fopen(path, "rb");
    if (!f) return;

    int32_t max_token_len = 0;
    fread(&max_token_len, sizeof(int32_t), 1, f);
    t->set_max_token_len(max_token_len);

    for (int i = 0; i < vocab_size; i++) {
        float score = 0.0f;
        int32_t len = 0;

        fread(&score, sizeof(float),   1, f);
        fread(&len,   sizeof(int32_t), 1, f);

        std::string token(len, '\0');
        fread(&token[0], 1, len, f);

        t->add_scores(score);
        t->add_vocab(token);
    }

    fclose(f);
}

// ---------------------------------------------------------------------------
// vocabLookup
// ---------------------------------------------------------------------------

static int vocabLookup(const TokenizerData* t, const std::string& str)
{
    for (int i = 0; i < t->vocab_sized(); i++)
        if (t->vocab(i) == str) return i;
    return -1;
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

static int encode(const TokenizerData* t, const char* text, int* tokens)
{
    int n = 0;
    std::string buf;

    // BOS token
    tokens[n++] = 1;

    // Single-character tokens
    for (const char* c = text; *c != '\0'; c++) {
        buf.assign(1, *c);
        int id = vocabLookup(t, buf);
        tokens[n++] = (id != -1) ? id : static_cast<unsigned char>(*c) + 3;
    }

    // Greedy BPE merges
    while (true) {
        float best    = -1e10f;
        int   best_id = -1, best_pos = -1;

        for (int i = 0; i < n - 1; i++) {
            buf = t->vocab(tokens[i]) + t->vocab(tokens[i + 1]);
            int id = vocabLookup(t, buf);
            if (id != -1 && t->scores(id) > best) {
                best     = t->scores(id);
                best_id  = id;
                best_pos = i;
            }
        }

        if (best_pos == -1) break;

        tokens[best_pos] = best_id;
        // Shift remaining tokens left
        for (int i = best_pos + 1; i < n - 1; i++)
            tokens[i] = tokens[i + 1];
        n--;
    }

    return n;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

static const char* decode(const TokenizerData* t, int prev, int cur)
{
    // vocab(cur) returns a const std::string& — take its c_str for compatibility.
    const char* s = t->vocab(cur).c_str();

    // Strip leading space after BOS token.
    if (prev == 1 && s[0] == ' ') s++;

    // Expand byte-encoded tokens of the form <0xXX>.
    unsigned char byte_val = 0;
    if (sscanf(s, "<0x%02hhX>", &byte_val) == 1) {
        static char byte_piece[2];
        byte_piece[0] = static_cast<char>(byte_val);
        byte_piece[1] = '\0';
        s = byte_piece;
    }

    return s;
}

// ---------------------------------------------------------------------------
// softmax overload for raw sub-array pointers (used inside forward())
// ---------------------------------------------------------------------------

static void softmax(float* x, int n)
{
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum  += x[i];
    }

    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

// Runs one token through the full transformer and returns a reference to the
// logits vector stored in transformer.state.  Callers must not cache the
// pointer across calls.
std::vector<float>& forward(Transformer& transformer, int token, int pos)
{
    const Config& config  = transformer.config;
    Weights&      weights = transformer.weights;
    RunState&     state   = transformer.state;

    const int dim       = config.dim();
    const int kv_dim    = (dim * config.n_kv_heads()) / config.num_heads();
    const int head_size = dim / config.num_heads();
    const int kv_mul    = config.num_heads() / config.n_kv_heads();
    const int hidden    = config.hidden_dim();

    // ---- Token embedding lookup ----------------------------------------
    // Copy the embedding row for this token into state.x.
    const float* emb = weights.token_embedding + token * dim;
    std::copy(emb, emb + dim, state.x.begin());

    // ---- Transformer layers --------------------------------------------
    for (int l = 0; l < config.num_layers(); l++) {

        // -- Attention pre-norm
        rmsnorm(state.xb, state.x, weights.rms_attention + l * dim, dim);

        // -- Q projection
        matmul(state.q, state.xb, weights.wQ + l * dim * dim, dim, dim);

        // -- RoPE: rotate Q in-place
        // Each pair (i, i+1) within every head gets the same rotation angle
        // derived from its position within the head (i % head_size).
        {
            float* q = state.q.data();
            for (int i = 0; i < dim; i += 2) {
                float freq  = 1.0f / powf(10000.0f, (i % head_size) / (float)head_size);
                float angle = pos * freq;
                float c_    = cosf(angle), s_ = sinf(angle);
                float v0 = q[i], v1 = q[i + 1];
                q[i]     = v0 * c_ - v1 * s_;
                q[i + 1] = v0 * s_ + v1 * c_;
            }
        }

        // -- K, V projections
        matmul(state.k, state.xb, weights.wK + l * dim * kv_dim, dim, kv_dim);
        matmul(state.v, state.xb, weights.wV + l * dim * kv_dim, dim, kv_dim);

        // -- RoPE: rotate K in-place (same formula, but over kv_dim)
        {
            float* k = state.k.data();
            for (int i = 0; i < kv_dim; i += 2) {
                float freq  = 1.0f / powf(10000.0f, (i % head_size) / (float)head_size);
                float angle = pos * freq;
                float c_    = cosf(angle), s_ = sinf(angle);
                float v0 = k[i], v1 = k[i + 1];
                k[i]     = v0 * c_ - v1 * s_;
                k[i + 1] = v0 * s_ + v1 * c_;
            }
        }

        // -- Write K and V into the KV-cache at this position
        const int loff   = l * config.seq_len() * kv_dim;
        float*    kc_pos = state.key_cache.data()   + loff + pos * kv_dim;
        float*    vc_pos = state.value_cache.data()  + loff + pos * kv_dim;
        std::copy(state.k.begin(), state.k.end(), kc_pos);
        std::copy(state.v.begin(), state.v.end(), vc_pos);

        // -- Multi-head attention
        for (int h = 0; h < config.num_heads(); h++) {
            float* q_h   = state.q.data()        + h * head_size;
            float* att_h = state.attention.data() + h * config.seq_len();
            float* xb_h  = state.xb.data()        + h * head_size;

            // Dot-product scores against all cached keys up to pos
            for (int t = 0; t <= pos; t++) {
                // GQA: multiple Q heads share one KV head (h / kv_mul)
                float* k_h  = state.key_cache.data()
                              + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                    score += q_h[i] * k_h[i];
                att_h[t] = score / sqrtf(static_cast<float>(head_size));
            }

            // Softmax over the (pos+1) scores for this head
            softmax(att_h, pos + 1);

            // Weighted sum of value vectors
            std::fill(xb_h, xb_h + head_size, 0.0f);
            for (int t = 0; t <= pos; t++) {
                float* v_h = state.value_cache.data()
                             + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_h[t];
                for (int i = 0; i < head_size; i++)
                    xb_h[i] += a * v_h[i];
            }
        }

        // -- Output projection + residual connection
        matmul(state.xb2, state.xb, weights.wO + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++)
            state.x[i] += state.xb2[i];

        // -- FFN pre-norm
        rmsnorm(state.xb, state.x, weights.rms_ffn + l * dim, dim);

        // -- FFN projections  (SwiGLU: gate = w1, up = w3, down = w2)
        matmul(state.hb,  state.xb, weights.w1 + l * dim * hidden, dim, hidden);
        matmul(state.hb2, state.xb, weights.w3 + l * dim * hidden, dim, hidden);

        // Element-wise SiLU(hb) * hb2
        for (int i = 0; i < hidden; i++) {
            float v    = state.hb[i];
            state.hb[i] = (v / (1.0f + expf(-v))) * state.hb2[i];
        }

        // -- FFN down-projection + residual connection
        matmul(state.xb, state.hb, weights.w2 + l * hidden * dim, hidden, dim);
        for (int i = 0; i < dim; i++)
            state.x[i] += state.xb[i];
    }

    // ---- Final norm + classifier projection ----------------------------
    // rmsnorm is safe in-place here: ss is computed from x before any write.
    rmsnorm(state.x, state.x, weights.rms_final, dim);
    matmul(state.logits, state.x, weights.wcls, dim, config.vocab_size());

    return state.logits;
}

} // namespace llama_infer