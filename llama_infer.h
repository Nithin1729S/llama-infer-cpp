#ifndef LLAMA_INFER_H
#define LLAMA_INFER_H

#include <vector>

void matmul(std::vector<float>& out,
            const std::vector<float>& x,
            const std::vector<float>& W,
            int n, int d);

void rmsnorm(std::vector<float>& out,
             const std::vector<float>& x,
             const std::vector<float>& w,
             int n);

void softmax(std::vector<float>& x, int n);

#endif