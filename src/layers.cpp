#include "layers.hpp"
#include <cmath>
#include <iostream>

// ============== LayerNorm ==============

ggml_tensor* LayerNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta
    // Assumes x is (seq_len, n_embd)
    ggml_tensor* mean = ggml_mean(ctx, x);
    ggml_tensor* x_centered = ggml_sub(ctx, x, mean);

    ggml_tensor* variance = ggml_sqr(ctx, x_centered);
    ggml_tensor* var_eps = ggml_add_float(ctx, variance, layer_norm_eps);
    ggml_tensor* std = ggml_sqrt(ctx, var_eps);

    ggml_tensor* normalized = ggml_div(ctx, x_centered, std);
    ggml_tensor* scaled = ggml_mul(ctx, normalized, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);

    return result;
}

// ============== RMSNorm ==============

ggml_tensor* RMSNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);
    ggml_tensor* var_eps = ggml_add_float(ctx, mean2, layer_norm_eps);
    ggml_tensor* rms = ggml_sqrt(ctx, var_eps);
    ggml_tensor* normalized = ggml_div(ctx, x, rms);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

// ============== GELU ==============

ggml_tensor* GELU::forward(ggml_context* ctx, ggml_tensor* x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);  // x^3
    ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, GELU_A));  // x + 0.044715 * x^3
    ggml_tensor* tanh_arg = ggml_scale(ctx, inner, GELU_SQRT2_OVER_PI);  // sqrt(2/pi) * inner
    ggml_tensor* tanh_result = ggml_tanh(ctx, tanh_arg);
    ggml_tensor* one_plus_tanh = ggml_add_float(ctx, tanh_result, 1.0f);
    ggml_tensor* result = ggml_scale(ctx, ggml_mul(ctx, x, one_plus_tanh), 0.5f);
    return result;
}

// ============== Attention ==============

Attention::Attention()
    : n_heads(GPT2Config::n_heads)
    , n_embd(GPT2Config::n_embd)
    , head_dim(GPT2Config::head_dim)
    , seq_len(0)
    , c_attn_weight(nullptr)
    , c_attn_bias(nullptr)
    , c_proj_weight(nullptr)
    , c_proj_bias(nullptr)
    , k_cache(nullptr)
    , v_cache(nullptr)
{}

void Attention::init_cache(ggml_context* ctx) {
    // KV cache shape: (n_heads, seq_len, head_dim)
    // We allocate max sequence length
    k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                  GPT2Config::head_dim,
                                  GPT2Config::n_heads,
                                  GPT2Config::context_length);
    v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                  GPT2Config::n_heads,
                                  GPT2Config::head_dim,
                                  GPT2Config::context_length);
    ggml_set_name(k_cache, "k_cache");
    ggml_set_name(v_cache, "v_cache");
}

ggml_tensor* Attention::causal_mask(ggml_context* ctx, int seq_len) {
    // Create causal mask: lower triangular matrix
    // mask[i,j] = 0 if j <= i (can attend), -inf if j > i (cannot attend)
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);

    float* data = (float*)mask->data;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            data[i * seq_len + j] = (j > i) ? -10000.0f : 0.0f;
        }
    }

    return mask;
}

ggml_tensor* Attention::forward(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    // x: (seq_len, n_embd)

    // QKV projection: compute q, k, v from input
    // c_attn_weight: (n_embd, 3 * n_embd), c_attn_bias: (3 * n_embd)
    ggml_tensor* qkv = ggml_linear(ctx, x, c_attn_weight, c_attn_bias);
    // qkv: (seq_len, 3 * n_embd)

    // Split into q, k, v
    // q: (seq_len, n_embd), k: (seq_len, n_embd), v: (seq_len, n_embd)
    int n_embd = GPT2Config::n_embd;
    ggml_tensor* q = ggml_view_2d(ctx, qkv, n_embd, 1, n_embd * sizeof(float), 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv, n_embd, 1, n_embd * sizeof(float), n_embd * sizeof(float));
    ggml_tensor* v = ggml_view_2d(ctx, qkv, n_embd, 1, n_embd * sizeof(float), 2 * n_embd * sizeof(float));

    // Reshape for multi-head: (seq_len, n_heads, head_dim)
    int n_heads = GPT2Config::n_heads;
    int head_dim = GPT2Config::head_dim;

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, 1);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, 1);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, 1);
    // q, k, v: (head_dim, n_heads, 1)

    // Transpose k and v for attention: (n_heads, head_dim, seq_len)
    k = ggml_transpose(ctx, k);
    v = ggml_transpose(ctx, v);

    // Store to KV cache if using cache
    if (use_cache && position >= 0) {
        // Copy k[:, :, 0] to k_cache[:, position, :]
        // This is a simplified approach - real implementation would use ggml_view
    }

    // Attention scores: q @ k^T / sqrt(head_dim)
    // q: (1, n_heads, head_dim), k: (n_heads, head_dim, seq_len)
    // scores: (1, n_heads, seq_len)
    ggml_tensor* scores = ggml_mul_mat(ctx, q, k);
    ggml_tensor* scaled_scores = ggml_scale_float(ctx, scores, 1.0f / std::sqrt(head_dim));

    // Apply causal mask
    ggml_tensor* mask = causal_mask(ctx, 1);  // seq_len = 1 for single token
    ggml_tensor* masked_scores = ggml_add(ctx, scaled_scores, mask);

    // Softmax
    ggml_tensor* attn_weights = ggml_softmax(ctx, masked_scores);

    // Apply attention to v: attn_weights @ v
    // attn_weights: (1, n_heads, seq_len), v: (n_heads, seq_len, head_dim)
    // result: (1, n_heads, head_dim)
    ggml_tensor* attn_out = ggml_mul_mat(ctx, attn_weights, v);
    // attn_out: (n_heads, head_dim, 1)

    // Reshape back to (1, n_embd)
    attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, 1);
    // attn_out: (1, n_embd)

    // Output projection
    ggml_tensor* out = ggml_linear(ctx, attn_out, c_proj_weight, c_proj_bias);
    // out: (1, n_embd)

    return out;
}

void Attention::set_weights(
    const float* qkv_w, const float* qkv_b,
    const float* proj_w, const float* proj_b
) {
    // This would copy weights to the GGML tensors
    // In practice, weights are loaded directly into tensors during model loading
}

// ============== FFN ==============

FFN::FFN()
    : c_fc_weight(nullptr)
    , c_fc_bias(nullptr)
    , c_proj_weight(nullptr)
    , c_proj_bias(nullptr)
{}

ggml_tensor* FFN::gelu(ggml_context* ctx, ggml_tensor* x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);
    ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, GELU_A));
    ggml_tensor* tanh_arg = ggml_scale(ctx, inner, GELU_SQRT2_OVER_PI);
    ggml_tensor* tanh_result = ggml_tanh(ctx, tanh_arg);
    ggml_tensor* one_plus_tanh = ggml_add_float(ctx, tanh_result, 1.0f);
    ggml_tensor* result = ggml_scale(ctx, ggml_mul(ctx, x, one_plus_tanh), 0.5f);
    return result;
}

ggml_tensor* FFN::forward(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x) {
    // FFN: GELU(up_proj(x)) * down_proj(x)

    // up_proj: (n_embd, n_ffn)
    ggml_tensor* up = ggml_linear(ctx, x, c_fc_weight, c_fc_bias);
    // up: (seq_len, n_ffn)

    // GELU activation
    ggml_tensor* activated = gelu(ctx, up);
    // activated: (seq_len, n_ffn)

    // down_proj: (n_ffn, n_embd)
    ggml_tensor* down = ggml_linear(ctx, activated, c_proj_weight, c_proj_bias);
    // down: (seq_len, n_embd)

    return down;
}

void FFN::set_weights(
    const float* fc_w, const float* fc_b,
    const float* proj_w, const float* proj_b
) {
    // Copy weights to tensors
}

// ============== TransformerBlock ==============

TransformerBlock::TransformerBlock() {}

ggml_tensor* TransformerBlock::forward(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    // Pre-norm architecture: LN1 -> Attention -> Residual
    ggml_tensor* ln1_out = layer_norm(ctx, x, ln1.gamma, ln1.beta, GPT2Config::layer_norm_eps);
    ggml_tensor* attn_out = attention.forward(ctx, gf, ln1_out, position, use_cache);
    ggml_tensor* h1 = ggml_add(ctx, x, attn_out);

    // LN2 -> FFN -> Residual
    ggml_tensor* ln2_out = layer_norm(ctx, h1, ln2.gamma, ln2.beta, GPT2Config::layer_norm_eps);
    ggml_tensor* ffn_out = ffn.forward(ctx, gf, ln2_out);
    ggml_tensor* h2 = ggml_add(ctx, h1, ffn_out);

    return h2;
}

void TransformerBlock::build_graph(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    // Pre-norm architecture
    ggml_tensor* ln1_out = layer_norm(ctx, x, ln1.gamma, ln1.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(gf, ln1_out);

    ggml_tensor* attn_out = attention.forward(ctx, gf, ln1_out, position, use_cache);
    ggml_build_forward_expand(gf, attn_out);

    ggml_tensor* h1 = ggml_add(ctx, x, attn_out);
    ggml_build_forward_expand(gf, h1);

    ggml_tensor* ln2_out = layer_norm(ctx, h1, ln2.gamma, ln2.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(gf, ln2_out);

    ggml_tensor* ffn_out = ffn.forward(ctx, gf, ln2_out);
    ggml_build_forward_expand(gf, ffn_out);

    ggml_tensor* h2 = ggml_add(ctx, h1, ffn_out);
    ggml_build_forward_expand(gf, h2);
}

// ============== Utilities ==============

ggml_tensor* linear(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias
) {
    ggml_tensor* result = ggml_mul_mat(ctx, input, weight);
    if (bias != nullptr) {
        result = ggml_add(ctx, result, bias);
    }
    return result;
}

ggml_tensor* layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* gamma,
    ggml_tensor* beta,
    float eps
) {
    ggml_tensor* mean = ggml_mean(ctx, x);
    ggml_tensor* x_centered = ggml_sub(ctx, x, mean);
    ggml_tensor* variance = ggml_sqr(ctx, x_centered);
    ggml_tensor* var_eps = ggml_add_float(ctx, variance, eps);
    ggml_tensor* std = ggml_sqrt(ctx, var_eps);
    ggml_tensor* normalized = ggml_div(ctx, x_centered, std);
    ggml_tensor* scaled = ggml_mul(ctx, normalized, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);
    return result;
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
) {
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);
    ggml_tensor* var_eps = ggml_add_float(ctx, mean2, eps);
    ggml_tensor* rms = ggml_sqrt(ctx, var_eps);
    ggml_tensor* normalized = ggml_div(ctx, x, rms);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}
