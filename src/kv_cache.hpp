#pragma once

#include <ggml.h>
#include <vector>
#include <cstring>

// KV Cache for efficient autoregressive generation
// Stores key and value activations for each layer to avoid recomputation

struct KVCacheEntry {
    ggml_tensor* k;  // Key tensor (n_heads, max_seq_len, head_dim)
    ggml_tensor* v;  // Value tensor (n_heads, max_seq_len, head_dim)

    int current_length;  // Number of tokens stored

    KVCacheEntry() : k(nullptr), v(nullptr), current_length(0) {}

    // Initialize cache tensors
    void init(ggml_context* ctx, int n_heads, int head_dim, int max_seq_len);

    // Store key/value for a new token at position
    void update(int position, const float* k_data, const float* v_data);

    // Get all keys/values up to position (for attention computation)
    void get(int position, float* k_out, float* v_out);
};

// Manages KV caches for all transformer layers
class KVCache {
public:
    static constexpr int N_LAYERS = 36;  // GPT-2 Large
    static constexpr int N_HEADS = 20;
    static constexpr int HEAD_DIM = 64;
    static constexpr int MAX_SEQ_LEN = 1024;

    KVCache();

    // Initialize all layer caches
    void init(ggml_context* ctx);

    // Reset all caches (clear all tokens)
    void reset();

    // Update cache with new token at given position
    void update(int layer_idx, int position, const float* k_data, const float* v_data);

    // Get pointer to layer's KV cache tensors
    KVCacheEntry& get_layer(int layer_idx) { return layers_[layer_idx]; }

    // Number of tokens currently cached
    int size() const { return current_length_; }

private:
    std::vector<KVCacheEntry> layers_;
    int current_length_;
};

// Helper to copy a slice of a 3D tensor
void copy_tensor_slice(
    ggml_tensor* dst,
    const ggml_tensor* src,
    int dst_offset,  // offset in the seq_len dimension
    int n_tokens     // number of tokens to copy
);
