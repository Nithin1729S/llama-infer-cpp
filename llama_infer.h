#ifndef LLAMA_INFER_H
#define LLAMA_INFER_H

#include <vector>
#include <memory>
#include <string>
#include "llama_model.pb.h"   // provides Config, TokenizerData

namespace llama_infer {

// ---------------------------------------------------------------------------
// Weight pointers — all raw pointers into LlamaModelData::buffer.
// The layout matches llama2.c export.py exactly (see wireWeights).
// ---------------------------------------------------------------------------
struct LlamaWeights {
    float* token_embedding = nullptr;
    float* rms_attention   = nullptr;
    float* wQ              = nullptr;
    float* wK              = nullptr;
    float* wV              = nullptr;
    float* wO              = nullptr;
    float* rms_ffn         = nullptr;
    float* w1              = nullptr;
    float* w2              = nullptr;
    float* w3              = nullptr;
    float* rms_final       = nullptr;
    float* wcls            = nullptr;   // == token_embedding when shared
};

// ---------------------------------------------------------------------------
// Owns the raw weight data loaded from the checkpoint file.
// ---------------------------------------------------------------------------
struct LlamaModelData {
    std::unique_ptr<float[]> buffer;
    size_t size = 0;          // bytes
};

// ---------------------------------------------------------------------------
// Intermediate activation tensors — sized once in allocState().
// ---------------------------------------------------------------------------
struct LlamaRunState {
    std::vector<float> x;           // current token embedding / residual stream
    std::vector<float> xb;          // scratch after rmsnorm / attention output
    std::vector<float> xb2;         // scratch after wO projection
    std::vector<float> q;           // query  [dim]
    std::vector<float> k;           // key    [kv_dim]
    std::vector<float> v;           // value  [kv_dim]
    std::vector<float> hb;          // FFN gate  [hidden_dim]
    std::vector<float> hb2;         // FFN up    [hidden_dim]
    std::vector<float> attention;   // [num_heads * seq_len]
    std::vector<float> logits;      // [vocab_size]
    std::vector<float> key_cache;   // [num_layers * seq_len * kv_dim]
    std::vector<float> value_cache; // [num_layers * seq_len * kv_dim]
};

// ---------------------------------------------------------------------------
// Top-level transformer bundle.
// ---------------------------------------------------------------------------
struct LlamaTransformer {
    Config         config;   // protobuf-generated
    LlamaWeights   weights;
    LlamaRunState  state;
    LlamaModelData data;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Load checkpoint binary and wire up weight pointers + run state.
void loadTransformer(LlamaTransformer& t, const char* path);

// Load SentencePiece-style tokenizer binary.
void loadTokenizer(TokenizerData* t, const char* path, int vocab_size);

// BPE encode: returns token count; tokens[0] is always BOS (1).
int encode(const TokenizerData* t, const char* text, int* tokens);

// Decode a single token id; handles leading-space stripping and byte pieces.
std::string decode(const TokenizerData* t, int prev, int cur);

// Run one transformer forward pass; returns reference to state.logits.
std::vector<float>& forward(LlamaTransformer& transformer, int token, int pos);

// Exposed helpers (used internally, also useful for testing).
void matmul(std::vector<float>& out,
            const std::vector<float>& x,
            const float* W, int n, int d);

void rmsnorm(std::vector<float>& out,
             const std::vector<float>& x,
             const float* w, int n);

void softmax(std::vector<float>& x, int n);

} // namespace llama_infer

#endif // LLAMA_INFER_H