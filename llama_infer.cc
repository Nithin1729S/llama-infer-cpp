#include <cmath>
#include <vector>
#include <memory>
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

void wireWeights(Weights* w, const Config& c, float* ptr, int shared)
{
    int dim = c.dim();
    int hidden = c.hidden_dim();
    int n_heads = c.num_heads();
    int n_kv = c.n_kv_heads();
    int seq_len = c.seq_len();
    int vocab = c.vocab_size();
    int layers = c.num_layers();

    int head_size = dim / n_heads;
    int kv_dim = (dim * n_kv) / n_heads;

    w->token_embedding = ptr;
    ptr += vocab * dim;

    w->rms_attention = ptr;
    ptr += layers * dim;

    w->wQ = ptr;
    ptr += layers * dim * dim;

    w->wK = ptr;
    ptr += layers * dim * kv_dim;

    w->wV = ptr;
    ptr += layers * dim * kv_dim;

    w->wO = ptr;
    ptr += layers * dim * dim;

    w->rms_ffn = ptr;
    ptr += layers * dim;

    w->w1 = ptr;
    ptr += layers * dim * hidden;

    w->w2 = ptr;
    ptr += layers * hidden * dim;

    w->w3 = ptr;
    ptr += layers * dim * hidden;

    w->rms_final = ptr;
    ptr += dim;

    ptr += seq_len * head_size;
    ptr += seq_len * head_size;

    if (shared)
        w->wcls = w->token_embedding;
    else {
        w->wcls = ptr;
        ptr += vocab * dim;
    }
}

} // namespace llama_infer