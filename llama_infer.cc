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

namespace {

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

} // anonymous namespace

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

void loadTransformer(ModelData& data,
                     Weights&   weights,
                     RunState&  state,
                     Config&    config,   // proto Config
                     const char* path)
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
    config.set_dim(bc.dim);
    config.set_hidden_dim(bc.hidden_dim);
    config.set_num_layers(bc.num_layers);
    config.set_num_heads(bc.num_heads);
    config.set_n_kv_heads(bc.n_kv_heads);
    config.set_vocab_size(std::abs(bc.vocab_size));
    config.set_seq_len(bc.seq_len);

    // 3. Memory-map the weight bytes.
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, sizeof(BinaryConfig), SEEK_SET);

    long wsize = file_size - static_cast<long>(sizeof(BinaryConfig));
    data.size   = static_cast<size_t>(wsize);
    data.buffer = std::make_unique<float[]>(wsize / sizeof(float));

    fread(data.buffer.get(), 1, wsize, file);
    fclose(file);

    wireWeights(&weights, config, data.buffer.get(), shared);
    allocState(&state, config);
}

// ---------------------------------------------------------------------------
// loadTokenizer
//
// Populates a proto TokenizerData message instead of a raw Tokenizer struct.
// ---------------------------------------------------------------------------

void loadTokenizer(TokenizerData* t, const char* path, int vocab_size)
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

int vocabLookup(const TokenizerData* t, const std::string& str)
{
    for (int i = 0; i < t->vocab_sized(); i++)
        if (t->vocab(i) == str) return i;
    return -1;
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

int encode(const TokenizerData* t, const char* text, int* tokens)
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

const char* decode(const TokenizerData* t, int prev, int cur)
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

} // namespace llama_infer