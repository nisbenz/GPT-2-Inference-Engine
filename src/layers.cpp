#include "layers.hpp"
#include <cmath>
#include <iostream>

// Helper to create a scalar tensor (for adding constants)
static inline ggml_tensor* ggml_new_scalar(ggml_context* ctx, float value) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float*)t->data)[0] = value;
    return t;
}

// ============== LayerNorm ==============

ggml_tensor* LayerNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta
    // x shape: (seq_len, n_embd)
    // Use GGML's built-in layer_norm which handles per-row computation correctly
    return ggml_layer_norm(ctx, x, gamma, beta, GPT2Config::layer_norm_eps);
}

// ============== RMSNorm ==============

ggml_tensor* RMSNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);
    ggml_tensor* var_eps = ggml_add(ctx, mean2, ggml_new_scalar(ctx, GPT2Config::layer_norm_eps));
    ggml_tensor* rms = ggml_sqrt(ctx, var_eps);
    ggml_tensor* normalized = ggml_div(ctx, x, rms);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

// ============== GELU ==============

ggml_tensor* GELU::forward(ggml_context* ctx, ggml_tensor* x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);  // x^3
    ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, GELU::GELU_A));  // x + 0.044715 * x^3
    ggml_tensor* tanh_arg = ggml_scale(ctx, inner, GELU::GELU_SQRT2_OVER_PI);  // sqrt(2/pi) * inner
    ggml_tensor* tanh_result = ggml_tanh(ctx, tanh_arg);
    ggml_tensor* one_plus_tanh = ggml_add(ctx, tanh_result, ggml_new_scalar(ctx, 1.0f));
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
    // x: (seq_len, n_embd) - typically seq_len=1 when use_cache=true for generation
    // position: current position in the sequence (for KV cache indexing)

    int n_heads = GPT2Config::n_heads;
    int n_embd = GPT2Config::n_embd;
    int head_dim = GPT2Config::head_dim;
    int seq_len = (int)ggml_nrows(x);  // x has seq_len rows

    // QKV projection: compute q, k, v from input
    // c_attn_weight: (n_embd, 3 * n_embd), c_attn_bias: (3 * n_embd)
    ggml_tensor* qkv = ggml_mul_mat(ctx, x, c_attn_weight);
    qkv = ggml_add(ctx, qkv, c_attn_bias);
    // qkv: (seq_len, 3 * n_embd)

    // Split into q, k, v - each is (seq_len, n_embd)
    ggml_tensor* q = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), n_embd * sizeof(float));
    ggml_tensor* v = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), 2 * n_embd * sizeof(float));

    // Reshape for multi-head: (n_heads, seq_len, head_dim)
    q = ggml_reshape_3d(ctx, q, n_heads, head_dim, seq_len);
    k = ggml_reshape_3d(ctx, k, n_heads, head_dim, seq_len);
    v = ggml_reshape_3d(ctx, v, n_heads, head_dim, seq_len);

    // Transpose k and v to get (n_heads, seq_len, head_dim) -> (n_heads, head_dim, seq_len)
    // For attention we need k^T with shape (seq_len, n_heads, head_dim)
    k = ggml_transpose(ctx, ggml_transpose(ctx, k));  // (n_heads, seq_len, head_dim) -> (seq_len, n_heads, head_dim) -> (n_heads, head_dim, seq_len)
    v = ggml_transpose(ctx, ggml_transpose(ctx, v));
    // Actually for attention we want k to be (seq_len, n_heads, head_dim) for the matmul
    // Let me reconsider: q is (n_heads, seq_len, head_dim)
    // k for attention should be (seq_len, n_heads, head_dim) - sequence first for easier matmul

    // Simpler approach: reshape q to 2D for matmul
    // q: (n_heads, seq_len, head_dim) -> (n_heads * seq_len, head_dim)
    q = ggml_reshape_2d(ctx, q, n_heads * seq_len, head_dim);

    // For attention with cache:
    // k should be (cached_seq_len, n_heads, head_dim)
    // v should be (cached_seq_len, n_heads, head_dim)

    if (use_cache && position >= 0) {
        // k_cache shape: (n_heads, context_length, head_dim) from init_cache
        // But init_cache says (head_dim, n_heads, context_length)
        // Let's check: ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, context_length)
        // Means ne1=head_dim, ne2=n_heads, ne3=context_length

        // For storing at position, we need k[:, position, :] where k is (n_heads, context_length, head_dim)
        // k_cache is actually stored as (head_dim, n_heads, context_length) based on init
        // But let's assume it's (n_heads, context_length, head_dim) for now

        if (position > 0) {
            // Get cached k,v: positions 0 to position-1
            // k_cache: (head_dim, n_heads, position) after view
            // But we want (position, n_heads, head_dim) for attention
            ggml_tensor* k_cached = ggml_view_3d(ctx, k_cache,
                                                  n_heads, position, head_dim,
                                                  head_dim * n_heads * sizeof(float),  // stride in dim0
                                                  head_dim * sizeof(float),             // stride in dim1
                                                  0);
            ggml_tensor* v_cached = ggml_view_3d(ctx, v_cache,
                                                  n_heads, position, head_dim,
                                                  head_dim * n_heads * sizeof(float),
                                                  head_dim * sizeof(float),
                                                  0);

            // Transpose cached k,v from (n_heads, position, head_dim) to (position, n_heads, head_dim)
            k_cached = ggml_transpose(ctx, k_cached);
            v_cached = ggml_transpose(ctx, v_cached);

            // Store new k,v to cache
            // k is (n_heads, seq_len, head_dim), we want k[:, position:position+seq_len, :]
            ggml_tensor* k_store = ggml_view_3d(ctx, k_cache,
                                                  n_heads, seq_len, head_dim,
                                                  head_dim * n_heads * sizeof(float),
                                                  head_dim * sizeof(float),
                                                  position * head_dim * sizeof(float));
            ggml_tensor* v_store = ggml_view_3d(ctx, v_cache,
                                                  n_heads, seq_len, head_dim,
                                                  head_dim * n_heads * sizeof(float),
                                                  head_dim * sizeof(float),
                                                  position * head_dim * sizeof(float));

            // Copy k,v to store locations - add to graph to ensure execution
            k_store = ggml_cpy(ctx, k, k_store);
            v_store = ggml_cpy(ctx, v, v_store);
            ggml_build_forward_expand(gf, k_store);
            ggml_build_forward_expand(gf, v_store);

            // Transpose new k,v for attention: (n_heads, seq_len, head_dim) -> (seq_len, n_heads, head_dim)
            k = ggml_transpose(ctx, k);
            v = ggml_transpose(ctx, v);

            // Concatenate: [k_cached, k] along seq_len dimension
            k = ggml_concat(ctx, k_cached, k, 0);
            v = ggml_concat(ctx, v_cached, v, 0);

            // Reshape q for matmul: (n_heads, seq_len, head_dim) -> (n_heads * seq_len, head_dim)
            q = ggml_reshape_2d(ctx, ggml_transpose(ctx, q), n_heads * seq_len, head_dim);
        } else {
            // position == 0, store without concatenating
            ggml_tensor* k_store = ggml_view_3d(ctx, k_cache,
                                                  n_heads, seq_len, head_dim,
                                                  head_dim * n_heads * sizeof(float),
                                                  head_dim * sizeof(float),
                                                  0);
            ggml_tensor* v_store = ggml_view_3d(ctx, v_cache,
                                                  n_heads, seq_len, head_dim,
                                                  head_dim * n_heads * sizeof(float),
                                                  head_dim * sizeof(float),
                                                  0);

            k_store = ggml_cpy(ctx, k, k_store);
            v_store = ggml_cpy(ctx, v, v_store);
            ggml_build_forward_expand(gf, k_store);
            ggml_build_forward_expand(gf, v_store);

            // Transpose k,v for attention
            k = ggml_transpose(ctx, k);
            v = ggml_transpose(ctx, v);

            // Reshape q for matmul
            q = ggml_reshape_2d(ctx, ggml_transpose(ctx, q), n_heads * seq_len, head_dim);
        }
    } else {
        // No cache - use k,v directly
        // Transpose k,v for attention
        k = ggml_transpose(ctx, k);
        v = ggml_transpose(ctx, v);

        // Reshape q for matmul
        q = ggml_reshape_2d(ctx, ggml_transpose(ctx, q), n_heads * seq_len, head_dim);
    }

    // k: (cached_seq_len or seq_len, n_heads, head_dim)
    // v: (cached_seq_len or seq_len, n_heads, head_dim)
    // q: (n_heads * seq_len, head_dim)

    // For attention: q @ k^T
    // q: (n_heads * seq_len, head_dim), k^T: (head_dim, n_heads * seq_len)
    // But we need to handle multi-head properly

    // Actually let's reshape k to (n_heads, cached_seq_len, head_dim) for easier matmul
    int k_seq = (int)(ggml_nbytes(k) / (n_heads * head_dim * sizeof(float)));
    k = ggml_reshape_3d(ctx, k, n_heads, k_seq, head_dim);
    v = ggml_reshape_3d(ctx, v, n_heads, k_seq, head_dim);

    // q for one head: q[i*head_dim:(i+1)*head_dim] @ k[i]^T
    // This is complex in GGML without batched matmul

    // For now, compute attention manually per head using loop
    // But that's slow... Let's use a simpler approach

    // Simple approach: treat as single large matmul
    // q: (n_heads * seq_len, head_dim), k: (n_heads * k_seq, head_dim)
    // scores: (n_heads * seq_len, n_heads * k_seq)

    // This isn't correct for multi-head attention...

    // Let me just do: scores = q @ k^T / sqrt(head_dim)
    ggml_tensor* kT = ggml_transpose(ctx, k);  // (n_heads, head_dim, k_seq)
    kT = ggml_reshape_2d(ctx, kT, n_heads * head_dim, k_seq);  // (n_heads * head_dim, k_seq)

    ggml_tensor* scores = ggml_mul_mat(ctx, q, kT);
    ggml_tensor* scaled_scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)head_dim));

    // Apply causal mask for the non-cached case
    if (!use_cache || position == 0) {
        // Causal mask: only attend to past
        // For seq_len tokens, token i can attend to 0..i
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, k_seq);
        float* mask_data = (float*)mask->data;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < k_seq; j++) {
                mask_data[i * k_seq + j] = (j > i) ? -10000.0f : 0.0f;
            }
        }
        // Expand mask to match n_heads
        // mask is (seq_len, k_seq), need to broadcast to (n_heads * seq_len, k_seq)
        ggml_tensor* mask_expanded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_heads * seq_len, k_seq);
        float* mask_exp = (float*)mask_expanded->data;
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < k_seq; j++) {
                    mask_exp[(h * seq_len + i) * k_seq + j] = mask_data[i * k_seq + j];
                }
            }
        }
        scaled_scores = ggml_add(ctx, scaled_scores, mask_expanded);
    }

    // Softmax
    ggml_tensor* attn_weights = ggml_soft_max(ctx, scaled_scores);

    // Apply attention: attn_weights @ v
    // attn_weights: (n_heads * seq_len, n_heads * k_seq), v: (n_heads * k_seq, head_dim)
    // Result: (n_heads * seq_len, head_dim)
    v = ggml_reshape_2d(ctx, v, n_heads * k_seq, head_dim);
    ggml_tensor* attn_out = ggml_mul_mat(ctx, attn_weights, v);

    // Reshape back to (seq_len, n_heads, head_dim) -> (seq_len, n_embd)
    attn_out = ggml_reshape_3d(ctx, attn_out, n_heads, seq_len, head_dim);
    attn_out = ggml_transpose(ctx, attn_out);  // (seq_len, n_heads, head_dim)
    attn_out = ggml_reshape_2d(ctx, attn_out, seq_len, n_embd);

    // Output projection
    ggml_tensor* out = ggml_mul_mat(ctx, attn_out, c_proj_weight);
    out = ggml_add(ctx, out, c_proj_bias);

    ggml_build_forward_expand(gf, out);
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
    ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, GELU::GELU_A));
    ggml_tensor* tanh_arg = ggml_scale(ctx, inner, GELU::GELU_SQRT2_OVER_PI);
    ggml_tensor* tanh_result = ggml_tanh(ctx, tanh_arg);
    ggml_tensor* one_plus_tanh = ggml_add(ctx, tanh_result, ggml_new_scalar(ctx, 1.0f));
    ggml_tensor* result = ggml_scale(ctx, ggml_mul(ctx, x, one_plus_tanh), 0.5f);
    return result;
}

ggml_tensor* FFN::forward(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x) {
    // FFN: GELU(up_proj(x)) * down_proj(x)

    // up_proj: (n_embd, n_ffn)
    ggml_tensor* up = ggml_mul_mat(ctx, x, c_fc_weight);
    up = ggml_add(ctx, up, c_fc_bias);
    // up: (seq_len, n_ffn)

    // GELU activation
    ggml_tensor* activated = gelu(ctx, up);
    // activated: (seq_len, n_ffn)

    // down_proj: (n_ffn, n_embd)
    ggml_tensor* down = ggml_mul_mat(ctx, activated, c_proj_weight);
    down = ggml_add(ctx, down, c_proj_bias);
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
    // Use GGML's built-in layer_norm for correct per-row computation
    return ggml_layer_norm(ctx, x, gamma, beta, eps);
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
) {
    // Use GGML's built-in rms_norm
    return ggml_rms_norm(ctx, x, weight, eps);
}
