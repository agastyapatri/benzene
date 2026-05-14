#include "tensor.hpp"
#include "nn.hpp"

int main(){

    // ── Test 1: LayerNorm ──────────────────────────────────
    bz::tensor x({2, 2, 4}, {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    bz::nn::LayerNorm ln({4}); // weight=1, bias=0 by default
    std::cout << "=== LayerNorm Output ===" << std::endl;
    std::cout << ln(x);

    // ── Test 2: SelfAttention ──────────────────────────────
    bz::tensor x2({1, 3, 3}, {
        1.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f
    });

    bz::tensor W_q({3, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    });

    bz::tensor W_k({3, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f
    });

    bz::tensor W_v({3, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    });

    bz::tensor keys    = bz::matmul(x2, W_k);
    bz::tensor queries = bz::matmul(x2, W_q);
    bz::tensor values  = bz::matmul(x2, W_v);

    bz::tensor attn_scores = bz::matmul(queries,
        bz::transpose(keys, keys.ndim()-1, keys.ndim()-2));
    bz::tensor attn_weights = bz::softmax(
        attn_scores * std::sqrtf(1.0f / 2), -1);
    bz::tensor context = bz::matmul(attn_weights, values);

    std::cout << "=== keys ===" << std::endl;    std::cout << keys;
    std::cout << "=== queries ===" << std::endl; std::cout << queries;
    std::cout << "=== values ===" << std::endl;  std::cout << values;
    std::cout << "=== attn_scores ===" << std::endl;  std::cout << attn_scores;
    std::cout << "=== attn_weights ===" << std::endl; std::cout << attn_weights;
    std::cout << "=== SelfAttention Output ===" << std::endl;
    std::cout << context;

    // ── Test 3: CausalSelfAttention ───────────────────────
    bz::tensor mask({3, 3}, {
         0.0f, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
         0.0f,  0.0f, -std::numeric_limits<float>::infinity(),
         0.0f,  0.0f,  0.0f
    });

    bz::tensor causal_scores  = attn_scores + mask;
    bz::tensor causal_weights = bz::softmax(
        causal_scores * std::sqrtf(1.0f / 2), -1);
    bz::tensor causal_context = bz::matmul(causal_weights, values);

    std::cout << "=== Causal attn_scores ===" << std::endl;  std::cout << causal_scores;
    std::cout << "=== Causal attn_weights ===" << std::endl; std::cout << causal_weights;
    std::cout << "=== CausalSelfAttention Output ===" << std::endl;
    std::cout << causal_context;

    return 0;
}
