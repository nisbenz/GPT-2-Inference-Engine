#include "model.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>

// ============== GPT2Model ==============

GPT2Model::GPT2Model()
    : ctx_(nullptr)
    , use_gpu_(false)
    , wte_(nullptr)
    , wpe_(nullptr)
    , lm_head_(nullptr)
    , logits_data_(nullptr)
    , logits_size_(0)
    , input_ids_tensor_(nullptr)
    , position_tensor_(nullptr)
{
    layers_.resize(N_LAYERS);
}

GPT2Model::~GPT2Model() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    if (logits_data_) {
        delete[] logits_data_;
    }
}

bool GPT2Model::init(bool use_gpu) {
    use_gpu_ = use_gpu;

    // Initialize GGML
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128 MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ctx_ = ggml_init(params);
    if (!ctx_) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return false;
    }

    // Initialize KV cache
    kv_cache_.init(ctx_);

    // Allocate model weights (will be filled by load_weights)
    // wte: (VOCAB_SIZE, N_EMBD) = (50257, 1280)
    wte_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, VOCAB_SIZE);
    ggml_set_name(wte_, "wte");
    memset(wte_->data, 0, ggml_nbytes(wte_));

    // wpe: (CONTEXT_LENGTH, N_EMBD) = (1024, 1280)
    wpe_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, CONTEXT_LENGTH);
    ggml_set_name(wpe_, "wpe");
    memset(wpe_->data, 0, ggml_nbytes(wpe_));

    // Final layer norm gamma and beta
    ln_f_.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
    ln_f_.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
    ggml_set_name(ln_f_.gamma, "ln_f_gamma");
    ggml_set_name(ln_f_.beta, "ln_f_beta");
    memset(ln_f_.gamma->data, 0, ggml_nbytes(ln_f_.gamma));
    memset(ln_f_.beta->data, 0, ggml_nbytes(ln_f_.beta));

    // LM head (tied to wte)
    lm_head_ = wte_;  // Tie weights

    // Initialize transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        // LayerNorm 1
        layers_[i].ln1.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        layers_[i].ln1.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(layers_[i].ln1.gamma, ("ln1_gamma_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ln1.beta, ("ln1_beta_" + std::to_string(i)).c_str());
        memset(layers_[i].ln1.gamma->data, 0, ggml_nbytes(layers_[i].ln1.gamma));
        memset(layers_[i].ln1.beta->data, 0, ggml_nbytes(layers_[i].ln1.beta));

        // Attention weights
        // c_attn_weight: (3 * N_EMBD, N_EMBD) = (3840, 1280)
        layers_[i].attention.c_attn_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, 3 * N_EMBD);
        layers_[i].attention.c_attn_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 3 * N_EMBD);
        // c_proj_weight: (N_EMBD, N_EMBD) = (1280, 1280)
        layers_[i].attention.c_proj_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, N_EMBD);
        layers_[i].attention.c_proj_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);

        ggml_set_name(layers_[i].attention.c_attn_weight, ("attn_c_attn_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_attn_bias, ("attn_c_attn_bias_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_proj_weight, ("attn_c_proj_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_proj_bias, ("attn_c_proj_bias_" + std::to_string(i)).c_str());

        memset(layers_[i].attention.c_attn_weight->data, 0, ggml_nbytes(layers_[i].attention.c_attn_weight));
        memset(layers_[i].attention.c_attn_bias->data, 0, ggml_nbytes(layers_[i].attention.c_attn_bias));
        memset(layers_[i].attention.c_proj_weight->data, 0, ggml_nbytes(layers_[i].attention.c_proj_weight));
        memset(layers_[i].attention.c_proj_bias->data, 0, ggml_nbytes(layers_[i].attention.c_proj_bias));

        // Initialize KV cache for this layer
        layers_[i].attention.init_cache(ctx_);

        // LayerNorm 2
        layers_[i].ln2.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        layers_[i].ln2.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(layers_[i].ln2.gamma, ("ln2_gamma_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ln2.beta, ("ln2_beta_" + std::to_string(i)).c_str());
        memset(layers_[i].ln2.gamma->data, 0, ggml_nbytes(layers_[i].ln2.gamma));
        memset(layers_[i].ln2.beta->data, 0, ggml_nbytes(layers_[i].ln2.beta));

        // FFN weights
        // c_fc_weight: (N_FFN, N_EMBD) = (5120, 1280)
        layers_[i].ffn.c_fc_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, N_FFN);
        layers_[i].ffn.c_fc_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_FFN);
        // c_proj_weight: (N_EMBD, N_FFN) = (1280, 5120)
        layers_[i].ffn.c_proj_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_FFN, N_EMBD);
        layers_[i].ffn.c_proj_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);

        ggml_set_name(layers_[i].ffn.c_fc_weight, ("ffn_c_fc_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_fc_bias, ("ffn_c_fc_bias_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_proj_weight, ("ffn_c_proj_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_proj_bias, ("ffn_c_proj_bias_" + std::to_string(i)).c_str());

        memset(layers_[i].ffn.c_fc_weight->data, 0, ggml_nbytes(layers_[i].ffn.c_fc_weight));
        memset(layers_[i].ffn.c_fc_bias->data, 0, ggml_nbytes(layers_[i].ffn.c_fc_bias));
        memset(layers_[i].ffn.c_proj_weight->data, 0, ggml_nbytes(layers_[i].ffn.c_proj_weight));
        memset(layers_[i].ffn.c_proj_bias->data, 0, ggml_nbytes(layers_[i].ffn.c_proj_bias));
    }

    // Initialize logits buffer
    logits_size_ = VOCAB_SIZE;
    logits_data_ = new float[logits_size_];
    memset(logits_data_, 0, logits_size_ * sizeof(float));

    std::cout << "GPT-2 Large model initialized" << std::endl;
    std::cout << "  Layers: " << N_LAYERS << std::endl;
    std::cout << "  Hidden: " << N_EMBD << std::endl;
    std::cout << "  Heads: " << N_HEADS << std::endl;
    std::cout << "  FFN: " << N_FFN << std::endl;
    std::cout << "  Vocab: " << VOCAB_SIZE << std::endl;

    return true;
}

bool GPT2Model::load_weights(const std::string& path) {
    // Try GGUF format first (most common for ggml-based models)
    if (path.find(".gguf") != std::string::npos ||
        path.find(".bin") != std::string::npos) {
        return load_gguf_weights(path);
    }
    // Fallback to legacy ggml format
    return load_ggml_weights(path);
}

bool GPT2Model::load_huggingface_weights(const std::string& path) {
    std::cout << "Loading HuggingFace weights from: " << path << std::endl;

    // This is a simplified loader
    // In reality, you'd use safetensors library or read PyTorch bin files
    // GPT-2 Large weights from HuggingFace:
    // - transformer.wte.weight: (50257, 1280)
    // - transformer.wpe.weight: (1024, 1280)
    // - transformer.h.{i}.ln_1.{gamma,beta}
    // - transformer.h.{i}.attn.{c_attn,c_proj}.{weight,bias}
    // - transformer.h.{i}.ln_2.{gamma,beta}
    // - transformer.h.{i}.mlp.{c_fc,c_proj}.{weight,bias}
    // - transformer.ln_f.{gamma,beta}
    // - lm_head.weight: (50257, 1280) - tied to wte

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weights file: " << path << std::endl;
        return false;
    }

    // Simplified: just show what we'd load
    std::cout << "Note: This is a stub - need actual weight loading implementation" << std::endl;
    std::cout << "For now, using random initialization" << std::endl;

    // Random initialization for testing
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Initialize with random weights for now
    float* wte_data = (float*)wte_->data;
    for (size_t i = 0; i < VOCAB_SIZE * N_EMBD; i++) {
        wte_data[i] = dist(gen);
    }

    float* wpe_data = (float*)wpe_->data;
    for (size_t i = 0; i < CONTEXT_LENGTH * N_EMBD; i++) {
        wpe_data[i] = dist(gen);
    }

    // Initialize layer weights
    for (int i = 0; i < N_LAYERS; i++) {
        float* ln1g = (float*)layers_[i].ln1.gamma->data;
        float* ln1b = (float*)layers_[i].ln1.beta->data;
        for (int j = 0; j < N_EMBD; j++) {
            ln1g[j] = (j == 0) ? 1.0f : 0.0f;  // gamma = 1
            ln1b[j] = 0.0f;  // beta = 0
        }

        float* c_attn_w = (float*)layers_[i].attention.c_attn_weight->data;
        float* c_attn_b = (float*)layers_[i].attention.c_attn_bias->data;
        float* c_proj_w = (float*)layers_[i].attention.c_proj_weight->data;
        float* c_proj_b = (float*)layers_[i].attention.c_proj_bias->data;

        for (size_t j = 0; j < 3 * N_EMBD * N_EMBD; j++) c_attn_w[j] = dist(gen);
        for (size_t j = 0; j < 3 * N_EMBD; j++) c_attn_b[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD * N_EMBD; j++) c_proj_w[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD; j++) c_proj_b[j] = dist(gen);

        float* ln2g = (float*)layers_[i].ln2.gamma->data;
        float* ln2b = (float*)layers_[i].ln2.beta->data;
        for (int j = 0; j < N_EMBD; j++) {
            ln2g[j] = (j == 0) ? 1.0f : 0.0f;
            ln2b[j] = 0.0f;
        }

        float* fc_w = (float*)layers_[i].ffn.c_fc_weight->data;
        float* fc_b = (float*)layers_[i].ffn.c_fc_bias->data;
        float* proj_w = (float*)layers_[i].ffn.c_proj_weight->data;
        float* proj_b = (float*)layers_[i].ffn.c_proj_bias->data;

        for (size_t j = 0; j < N_FFN * N_EMBD; j++) fc_w[j] = dist(gen);
        for (size_t j = 0; j < N_FFN; j++) fc_b[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD * N_FFN; j++) proj_w[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD; j++) proj_b[j] = dist(gen);
    }

    // Final layer norm
    float* ln_fg = (float*)ln_f_.gamma->data;
    float* ln_fb = (float*)ln_f_.beta->data;
    for (int j = 0; j < N_EMBD; j++) {
        ln_fg[j] = (j == 0) ? 1.0f : 0.0f;
        ln_fb[j] = 0.0f;
    }

    return true;
}

bool GPT2Model::load_gguf_weights(const std::string& path) {
    std::cout << "Loading GGUF model from: " << path << std::endl;

    // Load GGUF file
    struct ggml_context* meta_ctx = nullptr;
    struct gguf_context* gguf = gguf_load_from_file(path.c_str(), &meta_ctx);
    if (!gguf) {
        std::cerr << "Failed to load GGUF file: " << path << std::endl;
        return false;
    }

    // Get metadata
    int n_tensors = gguf_get_n_tensors(gguf);
    std::cout << "GGUF tensors: " << n_tensors << std::endl;

    // Print all tensor names for debugging
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf, i);
        printf("  tensor[%d]: %s\n", i, name);
    }

    // Helper to get tensor data and copy to our weights
    auto get_tensor_data = [&](const char* name, void* dest, size_t size) -> bool {
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
        if (!tensor) {
            // Try without prefix
            std::string alt_name = "transformer." + std::string(name);
            tensor = ggml_get_tensor(ctx_, alt_name.c_str());
        }
        if (!tensor) {
            std::cerr << "Warning: tensor not found: " << name << std::endl;
            return false;
        }
        if (ggml_nbytes(tensor) != size) {
            std::cerr << "Warning: tensor " << name << " size mismatch: "
                      << ggml_nbytes(tensor) << " != " << size << std::endl;
        }
        memcpy(dest, tensor->data, std::min(size, ggml_nbytes(tensor)));
        return true;
    };

    // Load token embeddings (wte): [vocab_size, n_embd]
    std::cout << "Loading wte..." << std::endl;
    float* wte_data = (float*)wte_->data;
    if (!get_tensor_data("wte.weight", wte_data, VOCAB_SIZE * N_EMBD * sizeof(float))) {
        // Try GPT-2 specific naming
        if (!get_tensor_data("transformer.wte.weight", wte_data, VOCAB_SIZE * N_EMBD * sizeof(float))) {
            std::cerr << "Failed to load wte.weight" << std::endl;
        }
    }

    // Load position embeddings (wpe): [context_length, n_embd]
    std::cout << "Loading wpe..." << std::endl;
    float* wpe_data = (float*)wpe_->data;
    if (!get_tensor_data("wpe.weight", wpe_data, CONTEXT_LENGTH * N_EMBD * sizeof(float))) {
        if (!get_tensor_data("transformer.wpe.weight", wpe_data, CONTEXT_LENGTH * N_EMBD * sizeof(float))) {
            std::cerr << "Failed to load wpe.weight" << std::endl;
        }
    }

    // Load transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        std::cout << "Loading layer " << i << "..." << std::endl;

        char name[128];

        // LayerNorm 1
        snprintf(name, sizeof(name), "h.%d.ln_1.weight", i);
        float* ln1g = (float*)layers_[i].ln1.gamma->data;
        get_tensor_data(name, ln1g, N_EMBD * sizeof(float));

        snprintf(name, sizeof(name), "h.%d.ln_1.bias", i);
        float* ln1b = (float*)layers_[i].ln1.beta->data;
        get_tensor_data(name, ln1b, N_EMBD * sizeof(float));

        // Attention QKV projection
        snprintf(name, sizeof(name), "h.%d.attn.c_attn.weight", i);
        float* c_attn_w = (float*)layers_[i].attention.c_attn_weight->data;
        get_tensor_data(name, c_attn_w, 3 * N_EMBD * N_EMBD * sizeof(float));

        snprintf(name, sizeof(name), "h.%d.attn.c_attn.bias", i);
        float* c_attn_b = (float*)layers_[i].attention.c_attn_bias->data;
        get_tensor_data(name, c_attn_b, 3 * N_EMBD * sizeof(float));

        // Attention output projection
        snprintf(name, sizeof(name), "h.%d.attn.c_proj.weight", i);
        float* c_proj_w = (float*)layers_[i].attention.c_proj_weight->data;
        get_tensor_data(name, c_proj_w, N_EMBD * N_EMBD * sizeof(float));

        snprintf(name, sizeof(name), "h.%d.attn.c_proj.bias", i);
        float* c_proj_b = (float*)layers_[i].attention.c_proj_bias->data;
        get_tensor_data(name, c_proj_b, N_EMBD * sizeof(float));

        // LayerNorm 2
        snprintf(name, sizeof(name), "h.%d.ln_2.weight", i);
        float* ln2g = (float*)layers_[i].ln2.gamma->data;
        get_tensor_data(name, ln2g, N_EMBD * sizeof(float));

        snprintf(name, sizeof(name), "h.%d.ln_2.bias", i);
        float* ln2b = (float*)layers_[i].ln2.beta->data;
        get_tensor_data(name, ln2b, N_EMBD * sizeof(float));

        // FFN up projection (c_fc)
        snprintf(name, sizeof(name), "h.%d.mlp.c_fc.weight", i);
        float* fc_w = (float*)layers_[i].ffn.c_fc_weight->data;
        get_tensor_data(name, fc_w, N_EMBD * N_FFN * sizeof(float));

        snprintf(name, sizeof(name), "h.%d.mlp.c_fc.bias", i);
        float* fc_b = (float*)layers_[i].ffn.c_fc_bias->data;
        get_tensor_data(name, fc_b, N_FFN * sizeof(float));

        // FFN down projection (c_proj)
        snprintf(name, sizeof(name), "h.%d.mlp.c_proj.weight", i);
        float* proj_w = (float*)layers_[i].ffn.c_proj_weight->data;
        get_tensor_data(name, proj_w, N_FFN * N_EMBD * sizeof(float));

        snprintf(name, sizeof(name), "h.%d.mlp.c_proj.bias", i);
        float* proj_b = (float*)layers_[i].ffn.c_proj_bias->data;
        get_tensor_data(name, proj_b, N_EMBD * sizeof(float));
    }

    // Final LayerNorm
    std::cout << "Loading ln_f..." << std::endl;
    float* ln_fg = (float*)ln_f_.gamma->data;
    float* ln_fb = (float*)ln_f_.beta->data;
    if (!get_tensor_data("ln_f.weight", ln_fg, N_EMBD * sizeof(float))) {
        get_tensor_data("transformer.ln_f.weight", ln_fg, N_EMBD * sizeof(float));
    }
    if (!get_tensor_data("ln_f.bias", ln_fb, N_EMBD * sizeof(float))) {
        get_tensor_data("transformer.ln_f.bias", ln_fb, N_EMBD * sizeof(float));
    }

    // LM head (tied to wte, so already loaded)
    // Some models have separate lm_head.weight
    float* lm_head_data = (float*)lm_head_->data;
    snprintf(name, sizeof(name), "lm_head.weight");
    if (!get_tensor_data(name, lm_head_data, VOCAB_SIZE * N_EMBD * sizeof(float))) {
        // Use wte as lm_head (tied weights) - already loaded
        std::cout << "Using tied lm_head (wte)" << std::endl;
    }

    gguf_free(gguf);
    if (meta_ctx) {
        ggml_free(meta_ctx);
    }

    std::cout << "GGUF weights loaded successfully!" << std::endl;
    return true;
}

bool GPT2Model::load_ggml_weights(const std::string& path) {
    std::cerr << "GGML weight format not yet implemented" << std::endl;
    return false;
}

std::vector<float> GPT2Model::forward(
    const std::vector<int>& input_ids,
    int position,
    bool use_cache
) {
    if (input_ids.empty()) {
        return std::vector<float>();
    }

    // Build computation graph
    build_graph(input_ids, position, use_cache);

    // Compute
    compute();

    // Return logits
    return std::vector<float>(logits_data_, logits_data_ + VOCAB_SIZE);
}

void GPT2Model::build_graph(
    const std::vector<int>& input_ids,
    int position,
    bool use_cache
) {
    int seq_len = input_ids.size();

    // Get token embeddings: wte[input_ids]
    // wte is (VOCAB_SIZE, N_EMBD), we need to index by input_ids
    // Result: (seq_len, N_EMBD)
    ggml_tensor* token_embeddings = ggml_get_rows(ctx_, wte_,
        ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, seq_len));

    // For now, simplify: use input_ids directly as indices
    // Create a view of wte with the selected rows
    ggml_tensor* input_embd = nullptr;
    for (int i = 0; i < seq_len; i++) {
        // Get embedding for token input_ids[i]
        ggml_tensor* row = ggml_view_2d(ctx_, wte_, N_EMBD, 1,
                                        N_EMBD * sizeof(float),
                                        input_ids[i] * N_EMBD * sizeof(float));
        if (i == 0) {
            input_embd = row;
        } else {
            input_embd = ggml_concat(ctx_, input_embd, row);
        }
    }
    // input_embd: (seq_len, N_EMBD)

    // Add positional embeddings
    ggml_tensor* pos_embd = nullptr;
    for (int i = 0; i < seq_len; i++) {
        int pos = position + i;
        if (pos >= CONTEXT_LENGTH) pos = CONTEXT_LENGTH - 1;  // Clip to max

        ggml_tensor* pos_row = ggml_view_2d(ctx_, wpe_, N_EMBD, 1,
                                            N_EMBD * sizeof(float),
                                            pos * N_EMBD * sizeof(float));
        if (i == 0) {
            pos_embd = pos_row;
        } else {
            pos_embd = ggml_concat(ctx_, pos_embd, pos_row);
        }
    }
    // pos_embd: (seq_len, N_EMBD)

    ggml_tensor* h = ggml_add(ctx_, input_embd, pos_embd);
    // h: (seq_len, N_EMBD)

    // Pass through transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        h = layers_[i].forward(ctx_, &gf_, h, position + i, use_cache);
        ggml_build_forward_expand(&gf_, h);
    }

    // Final layer norm
    ggml_tensor* h_norm = layer_norm(ctx_, h, ln_f_.gamma, ln_f_.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(&gf_, h_norm);

    // LM head: h_norm @ lm_head^T
    // lm_head is (VOCAB_SIZE, N_EMBD), we need (N_EMBD, VOCAB_SIZE)
    // Result: (seq_len, VOCAB_SIZE)
    ggml_tensor* logits = ggml_mul_mat(ctx_, h_norm, lm_head_);
    ggml_build_forward_expand(&gf_, logits);

    // Build the graph
    ggml_build_forward(&gf_, logits);
}

void GPT2Model::compute() {
    // Run computation
    ggml_graph_compute_with_ctx(ctx_, &gf_, 1);  // n_threads = 1

    // Extract logits from result
    ggml_tensor* logits_tensor = gf_.nodes[gf_.n_nodes - 1];
    if (logits_tensor && logits_tensor->data) {
        memcpy(logits_data_, logits_tensor->data,
               std::min((size_t)logits_size_, ggml_nbytes(logits_tensor)));
    }
}

std::vector<int> GPT2Model::generate(
    const std::string& prompt,
    int max_new_tokens,
    float temperature,
    int top_k
) {
    std::vector<int> tokens;

    // For now, just return EOS (placeholder for actual generation)
    // Real implementation would:
    // 1. Tokenize prompt
    // 2. Run forward pass
    // 3. Sample next token
    // 4. Append to tokens and repeat

    tokens.push_back(EOS_TOKEN);
    return tokens;
}

int GPT2Model::sample(const std::vector<float>& logits, float temperature, int top_k) {
    // Apply temperature
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
    }

    // Top-k filtering
    if (top_k > 0 && top_k < (int)probs.size()) {
        std::vector<std::pair<float, int>> pairs;
        for (int i = 0; i < (int)probs.size(); i++) {
            pairs.push_back({probs[i], i});
        }
        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        float top_k_sum = 0;
        for (int i = 0; i < top_k; i++) {
            top_k_sum += pairs[i].first;
        }
        for (int i = 0; i < top_k; i++) {
            probs[pairs[i].second] /= top_k_sum;
        }
        for (int i = top_k; i < (int)probs.size(); i++) {
            probs[pairs[i].second] = 0;
        }
    }

    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    return dist(gen);
}

// ============== Helper Functions ==============

ggml_tensor* create_tensor_2d(
    ggml_context* ctx,
    const std::string& name,
    int rows,
    int cols,
    const float* data
) {
    ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_set_name(tensor, name.c_str());
    if (data) {
        memcpy(tensor->data, data, ggml_nbytes(tensor));
    }
    return tensor;
}

ggml_tensor* create_tensor_1d(
    ggml_context* ctx,
    const std::string& name,
    int size,
    const float* data
) {
    ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    ggml_set_name(tensor, name.c_str());
    if (data) {
        memcpy(tensor->data, data, ggml_nbytes(tensor));
    }
    return tensor;
}

// ============== MappedFile ==============

MappedFile::MappedFile(const std::string& path) : data(nullptr), size(0) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        // Note: For production, use mmap or equivalent
        // This is simplified
        data = malloc(size);
        if (data) {
            file.read((char*)data, size);
        }
    }
}

MappedFile::~MappedFile() {
    if (data) {
        free(data);
    }
}
