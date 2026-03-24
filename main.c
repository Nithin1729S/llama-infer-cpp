#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    int dim;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float *token_embedding;
    float *rms_attention;
    float *wQ, *wK, *wV, *wO;
    float *w1, *w2, *w3;
    float *rms_ffn;
    float *rms_final;
    float *wcls;
} Weights;

typedef struct {
    float *x; 
    float *xb, *xb2;
    float *q, *k, *v;
    float *hb, *hb2;
    float *attention;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
    float *data;
} Transformer;

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_len;
} Tokenizer;

int main(int argc, char *argv[]) {
    Transformer transformer = {0};
    Tokenizer tokenizer = {0};
    return 0;
}