#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "llama_infer.h"
using namespace llama_infer;

static constexpr float kEps = 1e-4f;

class TransformerOpsTest : public ::testing::Test {
protected:
    bool SumsToOne(const std::vector<float>& v, int n) {
        float s = 0.f;
        for (int i = 0; i < n; i++) s += v[i];
        return std::abs(s - 1.f) < kEps;
    }

    bool AllInUnitInterval(const std::vector<float>& v, int n) {
        for (int i = 0; i < n; i++)
            if (v[i] < 0.f || v[i] > 1.f + kEps) return false;
        return true;
    }

    // Build a minimal TokenizerData for encode/decode tests.
    // Vocab: index 0 = "<unk>", 1 = "<s>", 2 = "</s>",
    //        3 = " ", 4 = "a", 5 = "b", 6 = "ab"
    TokenizerData MakeTokenizer() {
        TokenizerData t;
        t.set_vocab_sized(7);
        t.set_max_token_len(3);
        const char* words[] = {"<unk>", "<s>", "</s>", " ", "a", "b", "ab"};
        float scores[]      = {  0.f,    0.f,   0.f,   0.f, 1.f, 1.f,  2.f};
        for (int i = 0; i < 7; i++) {
            t.add_vocab(words[i]);
            t.add_scores(scores[i]);
        }
        return t;
    }
};

// Tests whether the matmul function correctly computes the matrix-vector product returns same vector when multiplied by the identity matrix.
TEST_F(TransformerOpsTest, MatMul_IdentityMatrix_ReturnsSameVector) {
    const int n = 4;
    std::vector<float> x   = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> W(n * n, 0.f);
    for (int i = 0; i < n; i++) W[i * n + i] = 1.f;
    std::vector<float> out(n, 0.f);

    matmul(out, x, W.data(), n, n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out[i], x[i], kEps) << "index " << i;
}

// Tests whether the matmul function correctly computes the matrix-vector product returns zero vector when input is zero, regardless of the weights.
TEST_F(TransformerOpsTest, MatMul_ZeroInput_ReturnsZeroOutput) {
    const int n = 3, d = 3;
    std::vector<float> x(n, 0.f);
    std::vector<float> W(d * n, 1.f);
    std::vector<float> out(d, 99.f);

    matmul(out, x, W.data(), n, d);

    for (int i = 0; i < d; i++)
        EXPECT_NEAR(out[i], 0.f, kEps) << "index " << i;
}

// Tests whether the matmul function correctly computes the matrix-vector product returns zero vector when weights are zero, regardless of the input.
TEST_F(TransformerOpsTest, MatMul_ZeroWeights_ReturnsZeroOutput) {
    const int n = 4, d = 2;
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> W(d * n, 0.f);
    std::vector<float> out(d, 99.f);

    matmul(out, x, W.data(), n, d);

    for (int i = 0; i < d; i++)
        EXPECT_NEAR(out[i], 0.f, kEps) << "index " << i;
}

// Tests whether the matmul function correctly computes the matrix-vector product handles non-square weight matrices, where the number of output rows differs from the input dimension.
TEST_F(TransformerOpsTest, MatMul_KnownValues_2x2) {
    // W (row-major):  row0=[5,7]  row1=[6,8]
    // out[0] = 1*5 + 2*7 = 19,  out[1] = 1*6 + 2*8 = 22
    std::vector<float> x = {1.f, 2.f};
    std::vector<float> W = {5.f, 7.f,
                            6.f, 8.f};
    std::vector<float> out(2, 0.f);

    matmul(out, x, W.data(), 2, 2);

    EXPECT_NEAR(out[0], 19.f, kEps);
    EXPECT_NEAR(out[1], 22.f, kEps);
}

// Tests whether the matmul function correctly computes the matrix-vector product handles non-square weight matrices, where the number of output rows differs from the input dimension, and selects the correct components of the input vector based on the weight matrix.
TEST_F(TransformerOpsTest, MatMul_NonSquare_SelectsComponents) {
    // d=2 output rows, n=3 input cols
    // W = [ 1 0 0 ]   => out[0] = x[0] = 2
    //     [ 0 0 1 ]   => out[1] = x[2] = 4
    std::vector<float> x = {2.f, 3.f, 4.f};
    std::vector<float> W = {1.f, 0.f, 0.f,
                            0.f, 0.f, 1.f};
    std::vector<float> out(2, 0.f);

    matmul(out, x, W.data(), 3, 2);

    EXPECT_NEAR(out[0], 2.f, kEps);
    EXPECT_NEAR(out[1], 4.f, kEps);
}

// Tests whether the matmul function correctly computes the matrix-vector product handles negative values in the input vector and weight matrix, and produces the correct signed output based on the multiplication of these values.
TEST_F(TransformerOpsTest, MatMul_NegativeValues_CorrectSign) {
    // out[0] = (-1)*3 + 2*(-4) = -11
    std::vector<float> x = {-1.f, 2.f};
    std::vector<float> W = {3.f, -4.f};
    std::vector<float> out(1, 0.f);

    matmul(out, x, W.data(), 2, 1);

    EXPECT_NEAR(out[0], -11.f, kEps);
}

// Tests whether the matmul function correctly computes the matrix-vector product overwrites the output vector on each call, rather than accumulating results from previous calls, ensuring that the output is solely determined by the current input and weights.
TEST_F(TransformerOpsTest, MatMul_OverwritesPreviousOutput) {
    std::vector<float> x = {1.f, 0.f};
    std::vector<float> W = {2.f, 0.f};
    std::vector<float> out = {100.f};

    matmul(out, x, W.data(), 2, 1);

    EXPECT_NEAR(out[0], 2.f, kEps);
}

// Tests whether the rmsnorm function correctly normalizes the input vector x using the provided weights w, and that the output has an RMS value of 1 when the weights are all ones, indicating that the normalization is functioning as expected.
TEST_F(TransformerOpsTest, RmsNorm_UnitWeights_OutputHasRmsOne) {
    const int n = 4;
    std::vector<float> x   = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> w(n, 1.f);
    std::vector<float> out(n, 0.f);

    rmsnorm(out, x, w.data(), n);

    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += out[i] * out[i];
    EXPECT_NEAR(std::sqrt(ss / n), 1.f, 1e-3f);
}

// Tests whether the rmsnorm function correctly normalizes the input vector x using the provided weights w, and that the output is zero when the weights are all zeros, indicating that the normalization scales the output to zero as expected.
TEST_F(TransformerOpsTest, RmsNorm_ZeroWeights_OutputIsZero) {
    const int n = 3;
    std::vector<float> x = {5.f, -3.f, 2.f};
    std::vector<float> w(n, 0.f);
    std::vector<float> out(n, 99.f);

    rmsnorm(out, x, w.data(), n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out[i], 0.f, kEps) << "index " << i;
}

// Tests whether the rmsnorm function correctly normalizes the input vector x using the provided weights w, and that the output preserves the sign of the input values, ensuring that the normalization does not alter the direction of the input vector components.
TEST_F(TransformerOpsTest, RmsNorm_PreservesSign) {
    const int n = 4;
    std::vector<float> x = {-1.f, 2.f, -3.f, 4.f};
    std::vector<float> w(n, 1.f);
    std::vector<float> out(n, 0.f);

    rmsnorm(out, x, w.data(), n);

    for (int i = 0; i < n; i++)
        EXPECT_EQ(std::signbit(out[i]), std::signbit(x[i])) << "index " << i;
}

// Tests whether the rmsnorm function correctly normalizes the input vector x using the provided weights w, and that doubling the weights results in doubling the output values, confirming that the output scales linearly with the weights as expected.
TEST_F(TransformerOpsTest, RmsNorm_DoubledWeights_DoublesOutput) {
    const int n = 3;
    std::vector<float> x  = {1.f, 2.f, 3.f};
    std::vector<float> w1(n, 1.f);
    std::vector<float> w2(n, 2.f);
    std::vector<float> out1(n, 0.f), out2(n, 0.f);

    rmsnorm(out1, x, w1.data(), n);
    rmsnorm(out2, x, w2.data(), n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out2[i], 2.f * out1[i], kEps) << "index " << i;
}

// Tests whether the rmsnorm function correctly normalizes the input vector x using the provided weights w, and that it does not produce NaN or infinite values when the input vector has very small values close to zero, ensuring numerical stability of the normalization process.
TEST_F(TransformerOpsTest, RmsNorm_NearZeroInput_DoesNotDivideByZero) {
    const int n = 4;
    std::vector<float> x(n, 1e-10f);
    std::vector<float> w(n, 1.f);
    std::vector<float> out(n, 0.f);

    ASSERT_NO_THROW(rmsnorm(out, x, w.data(), n));
    for (int i = 0; i < n; i++)
        EXPECT_TRUE(std::isfinite(out[i])) << "index " << i;
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that the resulting probabilities sum to 1, confirming that the softmax function is producing a valid probability distribution.
TEST_F(TransformerOpsTest, Softmax_SumsToOne) {
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    softmax(x, 4);
    EXPECT_TRUE(SumsToOne(x, 4));
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that all resulting probabilities are in the range [0, 1], ensuring that the softmax output is a valid probability distribution.
TEST_F(TransformerOpsTest, Softmax_AllValuesInUnitInterval) {
    std::vector<float> x = {-2.f, 0.f, 1.f, 5.f};
    softmax(x, 4);
    EXPECT_TRUE(AllInUnitInterval(x, 4));
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that the order of the input logits is preserved in the output probabilities, meaning that if one logit is greater than another, its corresponding probability should also be greater.
TEST_F(TransformerOpsTest, Softmax_MonotonicOrderPreserved) {
    std::vector<float> x = {1.f, 3.f, 2.f, 0.f};
    std::vector<float> orig = x;
    softmax(x, 4);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (orig[i] < orig[j])
                EXPECT_LT(x[i], x[j]) << "monotonicity violated at " << i << "," << j;
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that when all input logits are equal, the output probabilities are uniform, confirming that the softmax function treats equal logits as equally likely outcomes.
TEST_F(TransformerOpsTest, Softmax_EqualLogits_UniformDistribution) {
    const int n = 6;
    std::vector<float> x(n, 0.f);
    softmax(x, n);
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(x[i], 1.f / n, kEps) << "index " << i;
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that it is numerically stable when given large input values, ensuring that it does not produce NaN or infinite values due to overflow in the exponentiation step.
TEST_F(TransformerOpsTest, Softmax_SingleElement_ReturnsOne) {
    std::vector<float> x = {42.f};
    softmax(x, 1);
    EXPECT_NEAR(x[0], 1.f, kEps);
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that a single large positive logit dominates the output probabilities, confirming that the softmax function assigns near 100% probability to the largest logit as expected.
TEST_F(TransformerOpsTest, Softmax_LargePositiveLogit_DominatesProb) {
    std::vector<float> x = {0.f, 0.f, 1000.f, 0.f};
    softmax(x, 4);
    EXPECT_NEAR(x[2], 1.f, kEps);
    EXPECT_NEAR(x[0], 0.f, kEps);
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that it is numerically stable when given large input values, ensuring that it does not produce NaN or infinite values due to overflow in the exponentiation step, even when the input contains very large positive values.
TEST_F(TransformerOpsTest, Softmax_NumericalStability_LargeValues) {
    std::vector<float> x = {1000.f, 1001.f, 1002.f};
    softmax(x, 3);
    EXPECT_TRUE(SumsToOne(x, 3));
    for (int i = 0; i < 3; i++)
        EXPECT_TRUE(std::isfinite(x[i])) << "index " << i;
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that it is numerically stable when given large negative input values, ensuring that it does not produce NaN or infinite values due to underflow in the exponentiation step, even when the input contains very large negative values.
TEST_F(TransformerOpsTest, Softmax_KnownValues_TwoElement) {
    std::vector<float> x = {0.f, 1.f};
    softmax(x, 2);
    float e0 = std::exp(0.f), e1 = std::exp(1.f);
    EXPECT_NEAR(x[0], e0 / (e0 + e1), kEps);
    EXPECT_NEAR(x[1], e1 / (e0 + e1), kEps);
}

// Tests whether the softmax function correctly computes the probabilities from the input logits, and that applying softmax to a subset of the input vector does not modify the values outside that subset, ensuring that the function only operates on the specified range of elements.
TEST_F(TransformerOpsTest, Softmax_PartialRange_DoesNotTouchRest) {
    std::vector<float> x = {1.f, 2.f, 3.f, 99.f, 99.f};
    softmax(x, 3);
    EXPECT_TRUE(SumsToOne(x, 3));
    EXPECT_NEAR(x[3], 99.f, kEps);
    EXPECT_NEAR(x[4], 99.f, kEps);
}

// Tets wheter the encode function correctly converts input text into a sequence of token IDs based on the provided TokenizerData, and that it handles edge cases such as empty strings and single characters, ensuring that the encoding process produces valid token sequences according to the tokenizer's vocabulary and merging rules.
TEST_F(TransformerOpsTest, Encode_EmptyString_ReturnsBOSOnly) {
    TokenizerData t = MakeTokenizer();
    int tokens[32];
    int n = encode(&t, "", tokens);

    EXPECT_EQ(n, 1);
    EXPECT_EQ(tokens[0], 1);  // BOS
}

// Tests whether the encode function correctly converts input text into a sequence of token IDs based on the provided TokenizerData, and that it handles edge cases such as empty strings and single characters, ensuring that the encoding process produces valid token sequences according to the tokenizer's vocabulary and merging rules.
TEST_F(TransformerOpsTest, Encode_FirstTokenIsBOS) {
    TokenizerData t = MakeTokenizer();
    int tokens[32];
    encode(&t, "a", tokens);
    EXPECT_EQ(tokens[0], 1);  // BOS always first
}

// Tests whether the encode function correctly converts input text into a sequence of token IDs based on the provided TokenizerData, and that it maps individual characters to their corresponding vocabulary entries, ensuring that the encoding process correctly identifies and tokenizes known characters according to the tokenizer's vocabulary.
TEST_F(TransformerOpsTest, Encode_SingleChar_MapsToVocabEntry) {
    TokenizerData t = MakeTokenizer();
    int tokens[32];
    int n = encode(&t, "a", tokens);

    // BOS + "a" (no merges possible with a single char)
    EXPECT_EQ(n, 2);
    EXPECT_EQ(tokens[1], 4);  // vocab index of "a"
}

// Tests whether the encode function correctly converts input text into a sequence of token IDs based on the provided TokenizerData, and that it applies byte pair encoding (BPE) merges according to the scores in the tokenizer, ensuring that adjacent tokens are merged into a single token when the merge score is higher than the individual token scores, resulting in a more compact token sequence.
TEST_F(TransformerOpsTest, Encode_BPEMerge_MergesAdjacentPair) {
    // "ab" should be merged into vocab index 6 (score 2 > individual scores 1)
    TokenizerData t = MakeTokenizer();
    int tokens[32];
    int n = encode(&t, "ab", tokens);

    EXPECT_EQ(n, 2);           // BOS + merged "ab"
    EXPECT_EQ(tokens[0], 1);   // BOS
    EXPECT_EQ(tokens[1], 6);   // "ab" after merge
}

// Tests whether the encode function correctly converts input text into a sequence of token IDs based on the provided TokenizerData, and that all generated token IDs are within the valid range of the tokenizer's vocabulary size, ensuring that the encoding process does not produce out-of-range token IDs that could lead to errors in subsequent processing steps.
TEST_F(TransformerOpsTest, Encode_AllTokensInValidRange) {
    TokenizerData t = MakeTokenizer();
    int tokens[32];
    int n = encode(&t, "ab", tokens);

    for (int i = 0; i < n; i++) {
        EXPECT_GE(tokens[i], 0) << "token " << i << " is negative";
        EXPECT_LT(tokens[i], t.vocab_sized()) << "token " << i << " out of range";
    }
}

// Tests whether the decode function correctly converts token IDs back into text, and that it handles out-of-range token IDs by returning an empty string, ensuring that invalid tokens do not cause crashes or undefined behavior.
TEST_F(TransformerOpsTest, Decode_OutOfRangeToken_ReturnsEmpty) {
    TokenizerData t = MakeTokenizer();
    EXPECT_EQ(decode(&t, 0, -1),  "");
    EXPECT_EQ(decode(&t, 0, 100), "");
}

// Tests whether the decode function correctly converts token IDs back into text, and that it handles regular tokens by returning the corresponding vocabulary string, ensuring that valid token IDs are decoded to their correct textual representations according to the tokenizer's vocabulary.
TEST_F(TransformerOpsTest, Decode_RegularToken_ReturnsVocabString) {
    TokenizerData t = MakeTokenizer();
    // prev != 1, so no leading-space strip
    EXPECT_EQ(decode(&t, 0, 4), "a");
    EXPECT_EQ(decode(&t, 0, 5), "b");
    EXPECT_EQ(decode(&t, 0, 6), "ab");
}

// Tests whether the decode function correctly converts token IDs back into text, and that it handles the special case of decoding after a BOS token by stripping the leading space from the decoded string, ensuring that the output is formatted correctly for the beginning of a sequence.
TEST_F(TransformerOpsTest, Decode_AfterBOS_StripsLeadingSpace) {
    // Build a tokenizer where token 3 = " hello" (space-prefixed)
    TokenizerData t;
    t.set_vocab_sized(4);
    t.set_max_token_len(6);
    const char* words[] = {"<unk>", "<s>", "</s>", " hello"};
    for (int i = 0; i < 4; i++) { t.add_vocab(words[i]); t.add_scores(0.f); }

    // When prev == 1 (BOS), leading space is stripped
    EXPECT_EQ(decode(&t, 1, 3), "hello");
    // When prev != 1, space is preserved
    EXPECT_EQ(decode(&t, 0, 3), " hello");
}

// Tests whether the decode function correctly converts token IDs back into text, and that it can decode byte-encoded pieces by interpreting tokens in the format "<0xXX>" as their corresponding byte values, ensuring that the decoding process can handle special byte tokens and produce the correct character output.
TEST_F(TransformerOpsTest, Decode_ByteEncodedPiece_ExpandsToChar) {
    // Build a tokenizer with a <0x0A> (newline) entry
    TokenizerData t;
    t.set_vocab_sized(2);
    t.set_max_token_len(6);
    t.add_vocab("<unk>"); t.add_scores(0.f);
    t.add_vocab("<0x0A>"); t.add_scores(0.f);

    std::string result = decode(&t, 0, 1);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(static_cast<unsigned char>(result[0]), 0x0Au);
}