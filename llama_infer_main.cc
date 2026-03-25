#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "llama_infer.h"
#include "llama_model.pb.h"

using namespace llama_infer;

int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <model.bin> <tokenizer.bin> <prompt> [steps]\n",
                argv[0]);
        return 1;
    }

    LlamaTransformer transformer;
    TokenizerData    tokenizer;

    loadTransformer(transformer, argv[1]);
    loadTokenizer(&tokenizer, argv[2], transformer.config.vocab_size());

    // Encode prompt.
    int tokens[512];
    int n = encode(&tokenizer, argv[3], tokens);
    fprintf(stderr, "Encoded %d tokens\n", n);

    // Clamp steps the same way the reference C code does.
    int steps = (argc >= 5) ? atoi(argv[4]) : 256;
    if (steps > transformer.config.seq_len()) steps = transformer.config.seq_len();
    if (n     > transformer.config.seq_len()) n     = transformer.config.seq_len();
    if (steps < n) steps = n;

    int token = tokens[0];
    int prev  = 1;   // previous token; starts as BOS

    for (int pos = 0; pos < steps; pos++) {
        std::vector<float>& logits = forward(transformer, token, pos);

        // Greedy argmax decoding.
        int next = 0;
        for (int i = 1; i < transformer.config.vocab_size(); i++)
            if (logits[i] > logits[next]) next = i;

        if (pos < n - 1) {
            // Still feeding the prompt — advance through the token buffer.
            prev  = token;
            token = tokens[pos + 1];
        } else {
            // Generation phase.
            if (next == 1) break;   // EOS

            std::string piece = decode(&tokenizer, prev, next);
            printf("%s", piece.c_str());
            fflush(stdout);

            prev  = next;
            token = next;
        }
    }

    printf("\n");
    return 0;
}