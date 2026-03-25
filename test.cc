#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "main.h"

static constexpr float kEps = 1e-4f;   // tolerance for float comparisons
class TransformerOpsTest : public ::testing::Test {
protected:
    // Builds a simple identity-like weight matrix (row-major):
    //   W[i*n + j] = (i == j) ? 1.f : 0.f
    std::vector<float> IdentityMatrix(int n) {
        std::vector<float> W(n * n, 0.f);
        for (int i = 0; i < n; i++)
            W[i * n + i] = 1.f;
        return W;
    }

    // Builds a weight matrix filled with a constant value.
    std::vector<float> ConstantMatrix(int rows, int cols, float val) {
        return std::vector<float>(rows * cols, val);
    }

    // Checks whether a vector sums to ~1 (probability simplex).
    bool SumsToOne(const std::vector<float>& v, int n) {
        float s = 0.f;
        for (int i = 0; i < n; i++) s += v[i];
        return std::abs(s - 1.f) < kEps;
    }

    // Checks all elements are in [0, 1].
    bool AllInUnitInterval(const std::vector<float>& v, int n) {
        for (int i = 0; i < n; i++)
            if (v[i] < 0.f || v[i] > 1.f + kEps) return false;
        return true;
    }
};


TEST_F(TransformerOpsTest, MatMul_IdentityMatrix_ReturnsSameVector) {
    // Multiplying by the identity should leave the input unchanged.
    const int n = 4;
    std::vector<float> x   = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> W   = IdentityMatrix(n);
    std::vector<float> out(n, 0.f);

    matmul(out, x, W, n, n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out[i], x[i], kEps)
            << "Mismatch at index " << i;
}

TEST_F(TransformerOpsTest, MatMul_ZeroInput_ReturnsZeroOutput) {
    const int n = 3, d = 3;
    std::vector<float> x(n, 0.f);
    std::vector<float> W(d * n, 1.f);
    std::vector<float> out(d, 99.f);          // pre-fill with garbage

    matmul(out, x, W, n, d);

    for (int i = 0; i < d; i++)
        EXPECT_NEAR(out[i], 0.f, kEps) << "Index " << i;
}

TEST_F(TransformerOpsTest, MatMul_ZeroWeights_ReturnsZeroOutput) {
    const int n = 4, d = 2;
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> W(d * n, 0.f);
    std::vector<float> out(d, 99.f);

    matmul(out, x, W, n, d);

    for (int i = 0; i < d; i++)
        EXPECT_NEAR(out[i], 0.f, kEps) << "Index " << i;
}

TEST_F(TransformerOpsTest, MatMul_KnownValues_1x1) {
    // Scalar multiplication: out = x * W  (1×1)
    std::vector<float> x   = {3.f};
    std::vector<float> W   = {7.f};
    std::vector<float> out = {0.f};

    matmul(out, x, W, 1, 1);

    EXPECT_NEAR(out[0], 21.f, kEps);
}

TEST_F(TransformerOpsTest, MatMul_KnownValues_2x2) {
    // [ 1 2 ] * [ 5 6 ]T  =>  out[0] = 1*5 + 2*7 = 19
    //           [ 7 8 ]T      out[1] = 1*6 + 2*8 = 22
    std::vector<float> x = {1.f, 2.f};
    std::vector<float> W = {5.f, 7.f,   // row 0
                            6.f, 8.f};  // row 1
    std::vector<float> out(2, 0.f);

    matmul(out, x, W, 2, 2);

    EXPECT_NEAR(out[0], 19.f, kEps);
    EXPECT_NEAR(out[1], 22.f, kEps);
}

TEST_F(TransformerOpsTest, MatMul_NonSquare_NarrowInput) {
    // d=2 output rows, n=3 input cols
    // W = [ 1 0 0 ]   => out[0] = 1*x[0] = 2
    //     [ 0 0 1 ]   => out[1] = 1*x[2] = 4
    std::vector<float> x = {2.f, 3.f, 4.f};
    std::vector<float> W = {1.f, 0.f, 0.f,
                            0.f, 0.f, 1.f};
    std::vector<float> out(2, 0.f);

    matmul(out, x, W, 3, 2);

    EXPECT_NEAR(out[0], 2.f, kEps);
    EXPECT_NEAR(out[1], 4.f, kEps);
}

TEST_F(TransformerOpsTest, MatMul_ConstantWeights_SumsInput) {
    // All weights = 1 ⇒ out[i] = sum(x) for every i
    const int n = 5, d = 3;
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f, 5.f};
    float expected_sum = 15.f;
    std::vector<float> W  = ConstantMatrix(d, n, 1.f);
    std::vector<float> out(d, 0.f);

    matmul(out, x, W, n, d);

    for (int i = 0; i < d; i++)
        EXPECT_NEAR(out[i], expected_sum, kEps) << "Row " << i;
}

TEST_F(TransformerOpsTest, MatMul_NegativeValues_CorrectSign) {
    std::vector<float> x = {-1.f, 2.f};
    std::vector<float> W = {3.f, -4.f};   // 1-row × 2-col
    std::vector<float> out(1, 0.f);

    matmul(out, x, W, 2, 1);

    // out[0] = (-1)*3 + 2*(-4) = -3 - 8 = -11
    EXPECT_NEAR(out[0], -11.f, kEps);
}

TEST_F(TransformerOpsTest, MatMul_OutputOverwrites_NotAccumulates) {
    // Calling matmul twice should overwrite, not add to, previous output.
    std::vector<float> x = {1.f, 0.f};
    std::vector<float> W = {2.f, 0.f};
    std::vector<float> out = {100.f};        // pre-filled

    matmul(out, x, W, 2, 1);

    EXPECT_NEAR(out[0], 2.f, kEps);          // must overwrite 100
}

TEST_F(TransformerOpsTest, RmsNorm_UnitWeights_NormalisesVector) {
    // With w = all-ones, the output should have RMS == 1.
    const int n = 4;
    std::vector<float> x   = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> w(n, 1.f);
    std::vector<float> out(n, 0.f);

    rmsnorm(out, x, w, n);

    // Compute actual RMS of output
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += out[i] * out[i];
    float rms = std::sqrt(ss / n);

    EXPECT_NEAR(rms, 1.f, 1e-3f);
}

TEST_F(TransformerOpsTest, RmsNorm_ZeroWeights_OutputIsZero) {
    const int n = 3;
    std::vector<float> x = {5.f, -3.f, 2.f};
    std::vector<float> w(n, 0.f);
    std::vector<float> out(n, 99.f);

    rmsnorm(out, x, w, n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out[i], 0.f, kEps) << "Index " << i;
}

TEST_F(TransformerOpsTest, RmsNorm_PreservesSign) {
    const int n = 4;
    std::vector<float> x = {-1.f, 2.f, -3.f, 4.f};
    std::vector<float> w(n, 1.f);
    std::vector<float> out(n, 0.f);

    rmsnorm(out, x, w, n);

    // Signs must match input signs
    for (int i = 0; i < n; i++)
        EXPECT_EQ(std::signbit(out[i]), std::signbit(x[i]))
            << "Sign mismatch at index " << i;
}

TEST_F(TransformerOpsTest, RmsNorm_ScaledWeights_ScalesOutput) {
    // Doubling w should double the output exactly.
    const int n = 3;
    std::vector<float> x  = {1.f, 2.f, 3.f};
    std::vector<float> w1(n, 1.f);
    std::vector<float> w2(n, 2.f);
    std::vector<float> out1(n, 0.f), out2(n, 0.f);

    rmsnorm(out1, x, w1, n);
    rmsnorm(out2, x, w2, n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out2[i], 2.f * out1[i], kEps) << "Index " << i;
}

TEST_F(TransformerOpsTest, RmsNorm_SingleElement_CorrectNorm) {
    // With n=1: ss = x[0]^2; scale = 1/sqrt(x^2/1 + 1e-6) ≈ 1/|x|
    // out[0] ≈ x[0] * (1/|x|) * w[0] = sign(x) * w
    std::vector<float> x   = {4.f};
    std::vector<float> w   = {3.f};
    std::vector<float> out = {0.f};

    rmsnorm(out, x, w, 1);

    float expected = 4.f * (1.f / std::sqrt(16.f / 1.f + 1e-6f)) * 3.f;
    EXPECT_NEAR(out[0], expected, kEps);
}

TEST_F(TransformerOpsTest, RmsNorm_UniformInput_CorrectScale) {
    // x = [c, c, ..., c]  =>  ss/n = c^2  =>  scale = 1/sqrt(c^2+eps) ≈ 1/c
    // out[i] ≈ c * (1/c) * w[i] = w[i]
    const int   n = 5;
    const float c = 3.f;
    std::vector<float> x(n, c);
    std::vector<float> w = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float> out(n, 0.f);

    rmsnorm(out, x, w, n);

    float scale = 1.f / std::sqrt(c * c + 1e-6f);
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(out[i], c * scale * w[i], kEps) << "Index " << i;
}

TEST_F(TransformerOpsTest, RmsNorm_NumericalStability_NearZeroInput) {
    // Even with near-zero input, epsilon prevents division by zero.
    const int n = 4;
    std::vector<float> x(n, 1e-10f);
    std::vector<float> w(n, 1.f);
    std::vector<float> out(n, 0.f);

    ASSERT_NO_THROW(rmsnorm(out, x, w, n));

    for (int i = 0; i < n; i++)
        EXPECT_TRUE(std::isfinite(out[i])) << "Non-finite at index " << i;
}

TEST_F(TransformerOpsTest, Softmax_OutputSumsToOne) {
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    softmax(x, 4);
    EXPECT_TRUE(SumsToOne(x, 4));
}

TEST_F(TransformerOpsTest, Softmax_AllValuesInUnitInterval) {
    std::vector<float> x = {-2.f, 0.f, 1.f, 5.f};
    softmax(x, 4);
    EXPECT_TRUE(AllInUnitInterval(x, 4));
}

TEST_F(TransformerOpsTest, Softmax_MonotonicOrder_Preserved) {
    // Larger logit → larger probability.
    std::vector<float> x = {1.f, 3.f, 2.f, 0.f};
    std::vector<float> orig = x;
    softmax(x, 4);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (orig[i] < orig[j])
                EXPECT_LT(x[i], x[j])
                    << "Monotonicity violated for indices " << i << ", " << j;
}

TEST_F(TransformerOpsTest, Softmax_EqualLogits_UniformDistribution) {
    // All logits equal ⇒ each prob = 1/n.
    const int n = 6;
    std::vector<float> x(n, 0.f);
    softmax(x, n);

    for (int i = 0; i < n; i++)
        EXPECT_NEAR(x[i], 1.f / n, kEps) << "Index " << i;
}

TEST_F(TransformerOpsTest, Softmax_SingleElement_ReturnsOne) {
    std::vector<float> x = {42.f};
    softmax(x, 1);
    EXPECT_NEAR(x[0], 1.f, kEps);
}

TEST_F(TransformerOpsTest, Softmax_LargePositiveLogit_DominatesProb) {
    // One very large logit should take essentially all the probability.
    std::vector<float> x = {0.f, 0.f, 1000.f, 0.f};
    softmax(x, 4);
    EXPECT_NEAR(x[2], 1.f, kEps);
    EXPECT_NEAR(x[0], 0.f, kEps);
}

TEST_F(TransformerOpsTest, Softmax_NumericalStability_LargeValues) {
    // Without max-subtraction trick this would overflow; verify all finite.
    std::vector<float> x = {1000.f, 1001.f, 1002.f};
    softmax(x, 3);

    EXPECT_TRUE(SumsToOne(x, 3));
    for (int i = 0; i < 3; i++)
        EXPECT_TRUE(std::isfinite(x[i])) << "Non-finite at index " << i;
}

TEST_F(TransformerOpsTest, Softmax_NumericalStability_AllNegative) {
    std::vector<float> x = {-1000.f, -999.f, -998.f};
    softmax(x, 3);

    EXPECT_TRUE(SumsToOne(x, 3));
    for (int i = 0; i < 3; i++)
        EXPECT_TRUE(std::isfinite(x[i])) << "Non-finite at index " << i;
}

TEST_F(TransformerOpsTest, Softmax_KnownValues_TwoElement) {
    // softmax([0, 1]) = [e^0/(e^0+e^1), e^1/(e^0+e^1)]
    std::vector<float> x = {0.f, 1.f};
    softmax(x, 2);

    float e0 = std::exp(0.f), e1 = std::exp(1.f);
    EXPECT_NEAR(x[0], e0 / (e0 + e1), kEps);
    EXPECT_NEAR(x[1], e1 / (e0 + e1), kEps);
}

TEST_F(TransformerOpsTest, Softmax_InPlaceModification) {
    // Softmax is in-place; the original storage should be updated.
    std::vector<float> x = {2.f, 1.f, 0.f};
    float* ptr = x.data();

    softmax(x, 3);

    EXPECT_EQ(x.data(), ptr);      // same allocation
    EXPECT_TRUE(SumsToOne(x, 3));  // and values are valid probs
}

TEST_F(TransformerOpsTest, Softmax_PartialRange_DoesNotTouchRest) {
    // Call softmax on only the first 3 of 5 elements.
    std::vector<float> x = {1.f, 2.f, 3.f, 99.f, 99.f};
    softmax(x, 3);

    // First three must be valid probabilities
    EXPECT_TRUE(SumsToOne(x, 3));
    // Last two must be untouched
    EXPECT_NEAR(x[3], 99.f, kEps);
    EXPECT_NEAR(x[4], 99.f, kEps);
}