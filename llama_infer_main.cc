#include <iostream>
#include <vector>
#include "llama_infer.h"
#include "llama_model.pb.h"
using namespace llama_infer;
int main(){
    std::vector<float> x = {1.f, 2.f, 3.f};
    std::vector<float> W = {0.5f, 0.5f, 0.5f,
                            0.5f, 0.5f, 0.5f};
    std::vector<float> out(2, 0.f);

    matmul(out, x, W, 3, 2);

    std::cout << "Output: ";
    for (float val : out)
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}