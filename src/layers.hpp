#pragma once

#include <ggml/ggml.h>
#include <vector>
#include <string>

// GPT-2 Large Configuration: 36 layers, 1280 hidden, 20 heads, 5120 FFN intermediate
struct GPT2Config {
    static constexpr int n_layers = 36;
    static constexpr int n_heads = 20;
    static constexpr int n_embd = 1280;       // Hidden size
    static constexpr int n_ffn = 5120;        // FFN intermediate size
    static constexpr int vocab_size = 50257;
    static constexpr int context_length = 1024;
    static constexpr int head_dim = n_embd / n_heads;  // 64
    static constexpr float layer_norm_eps = 1e-5f;
};

// Layer Normalization (pre-norm style: gamma * (x - mean) / sqrt(var + eps) + beta)
struct LayerNorm {
    ggml_tensor* gamma;  // scale
    ggml_tensor* beta;   // bias

    LayerNorm() : gamma(nullptr), beta(nullptr) {}

    // Forward pass: output = gamma * (x - mean) / sqrt(var + eps) + beta
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
};

// RMSNorm (more common in modern transformers, simpler)
struct RMSNorm {
    ggml_tensor* weight;  // gamma

    RMSNorm() : weight(nullptr) {}

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
};

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
struct GELU {
    static constexpr float GELU_A = 0.044715f;
    static constexpr float GELU_SQRT2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
};

// Multi-Head Causal Self-Attention
struct Attention {
    // Projection weights (stored as GGML tensors)
    ggml_tensor* c_attn_weight;  // qkv_proj: (n_embd, 3 * n_embd)
    ggml_tensor* c_attn_bias;    // qkv_proj bias: (3 * n_embd)
    ggml_tensor* c_proj_weight;  // output_proj: (n_embd, n_embd)
    ggml_tensor* c_proj_bias;    // output_proj bias: (n_embd)

    // KV Cache for this layer
    ggml_tensor* k_cache;  // (n_heads, seq_len, head_dim)
    ggml_tensor* v_cache;  // (n_heads, seq_len, head_dim)

    int n_heads;
    int n_embd;
    int head_dim;
    int seq_len;

    Attention();

    // Initialize KV cache tensors
    void init_cache(ggml_context* ctx);

    // Causal mask: prevent attending to future tokens
    // token at position i can only attend to positions 0..i
    static ggml_tensor* causal_mask(ggml_context* ctx, int seq_len);

    // Multi-head attention forward
    // x: input tensor (seq_len, n_embd)
    // position: current position in sequence
    // use_cache: whether to use/store KV cache
    ggml_tensor* forward(
        ggml_context* ctx,
        ggml_cgraph* gf,
        ggml_tensor* x,
        int position,
        bool use_cache = true
    );

    // Set weights from loaded model
    void set_weights(
        const float* qkv_w, const float* qkv_b,
        const float* proj_w, const float* proj_b
    );
};

// Feed-Forward Network: GELU(up_proj(x)) * down_proj(x)
struct FFN {
    ggml_tensor* c_fc_weight;   // up_proj: (n_embd, n_ffn)
    ggml_tensor* c_fc_bias;       // up_proj bias: (n_ffn)
    ggml_tensor* c_proj_weight;   // down_proj: (n_ffn, n_embd)
    ggml_tensor* c_proj_bias;     // down_proj bias: (n_embd)

    FFN();

    // GELU activation
    static ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x);

    // Forward pass
    ggml_tensor* forward(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x);

    // Set weights
    void set_weights(
        const float* fc_w, const float* fc_b,
        const float* proj_w, const float* proj_b
    );
};

// Transformer Block: LN1 -> Attention -> Residual -> LN2 -> FFN -> Residual
struct TransformerBlock {
    LayerNorm ln1;
    LayerNorm ln2;
    Attention attention;
    FFN ffn;

    TransformerBlock();

    // Forward pass through one transformer block
    // x: input (seq_len, n_embd)
    // position: current position
    // use_cache: whether to use KV cache
    ggml_tensor* forward(
        ggml_context* ctx,
        ggml_cgraph* gf,
        ggml_tensor* x,
        int position,
        bool use_cache = true
    );

    // Build computation graph for this block
    void build_graph(
        ggml_context* ctx,
        ggml_cgraph* gf,
        ggml_tensor* x,
        int position,
        bool use_cache = true
    );
};

// Utility: Create a linear layer tensor operation
ggml_tensor* linear(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias
);

// Utility: Create LayerNorm operation
ggml_tensor* layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* gamma,
    ggml_tensor* beta,
    float eps
);

// Utility: Create RMSNorm operation
ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
);
