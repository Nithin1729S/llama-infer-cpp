#include <iostream>
#include <vector>
#include <cmath>
#include "model.pb.h"

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

struct Transformer {
    Config config;     
    Weights weights;   
    RunState state;
};

struct TokenizerRuntime {
    std::vector<std::string> vocab;
    std::vector<float> scores;
    int max_token_len;
};

static void matmul(float* out, const float* x, const float* W, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0;
        for (int j = 0; j < n; j++)
            val += x[j] * W[i * n + j];
        out[i] = val;
    }
}

static void rmsnorm(float* out, const float* x, const float* w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * ss * w[i];
}

static void softmax(float* x, int n) {
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

int main() {
    Transformer transformer;
    Tokenizer Tokenizer;
    return 0;
}