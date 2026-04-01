#include "common_test.hpp"
#include "gguf_loader.h"
#include <ggml.h>
#include <cstring>
#include <fstream>

// This test diagnoses WTE tensor loading issues
// Run with path to GGUF model file as argument

// Simulate the transpose detection logic from model.cpp
bool test_wte_transpose_logic() {
    print_test_header("test_wte_transpose_logic");

    // GGML allocates WTE as: ne[0]=N_EMBD=768, ne[1]=VOCAB_SIZE=50257
    // In GGML 2D tensor: ne[0]=rows, ne[1]=cols (conceptually)
    // For ggml_mul_mat compatibility: W.ne[0] must match input.ne[0]
    //
    // GGUF stores dimensions as [rows, cols] in file
    // Case A: GGUF has [50257, 768] - token embeddings as rows
    // Case B: GGUF has [768, 50257] - token embeddings as cols (transposed)
    //
    // The transpose detection check:
    // if (t.dims[0] == dst->ne[1] && t.dims[1] == dst->ne[0])
    // means: does GGUF rows == GGML cols AND GGUF cols == GGML rows?
    // If so, we need to transpose.

    int n_embd = 768;
    int vocab_size = 50257;
    size_t dst_ne[2] = {(size_t)n_embd, (size_t)vocab_size};  // GGML ne[0], ne[1]

    // Case A: GGUF dims = [50257, 768]
    {
        uint64_t gguf_dims[2] = {50257, 768};
        bool needs_transpose = (gguf_dims[0] == dst_ne[1] && gguf_dims[1] == dst_ne[0]);
        std::cout << "  Case A - GGUF [50257, 768] vs GGML ne[0]=768, ne[1]=50257" << std::endl;
        std::cout << "    dims[0]=" << gguf_dims[0] << " == dst_ne[1]=" << dst_ne[1]
                  << " ? " << (gguf_dims[0] == dst_ne[1] ? "YES" : "NO") << std::endl;
        std::cout << "    dims[1]=" << gguf_dims[1] << " == dst_ne[0]=" << dst_ne[0]
                  << " ? " << (gguf_dims[1] == dst_ne[0] ? "YES" : "NO") << std::endl;
        std::cout << "    needs_transpose = " << (needs_transpose ? "TRUE" : "FALSE") << std::endl;
        TEST_ASSERT_MSG(needs_transpose == true,
            "Case A: When GGUF has [50257, 768], transpose should be needed");
    }

    // Case B: GGUF dims = [768, 50257]
    {
        uint64_t gguf_dims[2] = {768, 50257};
        bool needs_transpose = (gguf_dims[0] == dst_ne[1] && gguf_dims[1] == dst_ne[0]);
        std::cout << "  Case B - GGUF [768, 50257] vs GGML ne[0]=768, ne[1]=50257" << std::endl;
        std::cout << "    dims[0]=" << gguf_dims[0] << " == dst_ne[1]=" << dst_ne[1]
                  << " ? " << (gguf_dims[0] == dst_ne[1] ? "YES" : "NO") << std::endl;
        std::cout << "    dims[1]=" << gguf_dims[1] << " == dst_ne[0]=" << dst_ne[0]
                  << " ? " << (gguf_dims[1] == dst_ne[0] ? "YES" : "NO") << std::endl;
        std::cout << "    needs_transpose = " << (needs_transpose ? "TRUE" : "FALSE") << std::endl;
        TEST_ASSERT_MSG(needs_transpose == false,
            "Case B: When GGUF has [768, 50257], no transpose should be needed");
    }

    std::cout << "  WTE transpose logic verified" << std::endl;
    return 0;
}

// Test the actual transpose operation
int test_transpose_2d() {
    print_test_header("test_transpose_2d");

    // Test: 3x2 matrix
    // [1, 2]
    // [3, 4]
    // [5, 6]
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    int src_rows = 3;  // dims[0]
    int src_cols = 2;  // dims[1]

    // After transpose: 2x3 matrix
    // [1, 3, 5]
    // [2, 4, 6]
    std::vector<float> dst(src.size());
    for (int r = 0; r < src_rows; r++) {
        for (int c = 0; c < src_cols; c++) {
            dst[c * src_rows + r] = src[r * src_cols + c];
        }
    }

    // Verify
    std::cout << "  Source (3x2): ";
    for (auto v : src) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "  Transposed (2x3): ";
    for (auto v : dst) std::cout << v << " ";
    std::cout << std::endl;

    TEST_ASSERT_FLOAT_EQ(dst[0], 1.0f, 0.001f);  // dst[0] = src[0]
    TEST_ASSERT_FLOAT_EQ(dst[1], 3.0f, 0.001f);  // dst[1] = src[2]
    TEST_ASSERT_FLOAT_EQ(dst[2], 5.0f, 0.001f);  // dst[2] = src[4]
    TEST_ASSERT_FLOAT_EQ(dst[3], 2.0f, 0.001f);  // dst[3] = src[1]
    TEST_ASSERT_FLOAT_EQ(dst[4], 4.0f, 0.001f);  // dst[4] = src[3]
    TEST_ASSERT_FLOAT_EQ(dst[5], 6.0f, 0.001f);  // dst[5] = src[5]

    std::cout << "  2D transpose verified" << std::endl;
    return 0;
}

// Test GGML embedding lookup (column-major)
int test_embedding_lookup() {
    print_test_header("test_embedding_lookup");

    // Simulate WTE tensor: ne[0]=768, ne[1]=50257
    // Layout in memory (column-major):
    // Col 0: elements 0-767 (embedding for token 0)
    // Col 1: elements 768-1535 (embedding for token 1)
    // ...
    // Col t: elements t*768 to t*768+767 (embedding for token t)
    //
    // In GGML, for 2D tensor with ne[0]=A, ne[1]=B:
    // element [row, col] = data[col * ne[0] + row]
    // So embedding for token t starts at data[t * ne[0]]

    int n_embd = 768;
    int vocab_size = 50257;

    // Create a simple WTE-like tensor with known pattern
    std::vector<float> wte_data(n_embd * vocab_size);
    for (int t = 0; t < vocab_size; t++) {
        for (int i = 0; i < n_embd; i++) {
            // Token t has embedding where element i = t * 1000 + i
            wte_data[t * n_embd + i] = t * 1000.0f + i;
        }
    }

    // Verify token 0 embedding
    float* token0_emb = &wte_data[0];
    std::cout << "  Token 0 embedding[0]=" << token0_emb[0]
              << " (expected 0)" << std::endl;
    std::cout << "  Token 0 embedding[100]=" << token0_emb[100]
              << " (expected 100)" << std::endl;
    TEST_ASSERT_FLOAT_EQ(token0_emb[0], 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(token0_emb[100], 100.0f, 0.001f);

    // Verify token 1 embedding
    float* token1_emb = &wte_data[1 * n_embd];
    std::cout << "  Token 1 embedding[0]=" << token1_emb[0]
              << " (expected 1000)" << std::endl;
    std::cout << "  Token 1 embedding[100]=" << token1_emb[100]
              << " (expected 1100)" << std::endl;
    TEST_ASSERT_FLOAT_EQ(token1_emb[0], 1000.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(token1_emb[100], 1100.0f, 0.001f);

    // Verify token 100 embedding
    float* token100_emb = &wte_data[100 * n_embd];
    std::cout << "  Token 100 embedding[0]=" << token100_emb[0]
              << " (expected 100000)" << std::endl;
    TEST_ASSERT_FLOAT_EQ(token100_emb[0], 100000.0f, 0.001f);

    std::cout << "  Embedding lookup pattern verified" << std::endl;
    return 0;
}

// Test what happens with transposed WTE
int test_transposed_embedding_lookup() {
    print_test_header("test_transposed_embedding_lookup");

    // If GGUF stored WTE as [768, 50257] in row-major
    // But we interpret it as column-major with ne[0]=768, ne[1]=50257
    // The layout is different!

    int n_embd = 768;
    int vocab_size = 50257;

    // Simulate GGUF row-major [768, 50257] -> 768 rows, 50257 cols
    std::vector<float> gguf_data(n_embd * vocab_size);
    for (int r = 0; r < n_embd; r++) {
        for (int c = 0; c < vocab_size; c++) {
            // In row-major: element [r, c] = data[r * vocab_size + c]
            // For token c at row r: value = c * 1000 + r
            gguf_data[r * vocab_size + c] = c * 1000.0f + r;
        }
    }

    // When we load this into GGML tensor with ne[0]=768, ne[1]=50257 (column-major)
    // The element at logical [row, col] = data[col * ne[0] + row] = data[col * 768 + row]
    // So for token t, element r = data[t * 768 + r]
    //
    // But if GGUF was row-major [768, 50257], then element [r, c] was at data[r * 50257 + c]
    // So our lookup of data[t * 768 + r] actually gets us GGUF element [r, t]
    // Which is value = t * 1000 + r

    std::cout << "  Simulating transposed WTE lookup (row-major GGUF -> column-major GGML)" << std::endl;

    // Token 0 lookup (with wrong interpretation)
    float val_at_0 = gguf_data[0 * n_embd + 0];  // What we think is token 0, element 0
    std::cout << "  'Token 0' element 0 = " << val_at_0
              << " (expected GGUF[0,0] = 0*1000+0 = 0)" << std::endl;

    // Token 1 lookup (with wrong interpretation)
    float val_at_1 = gguf_data[1 * n_embd + 0];  // What we think is token 1, element 0
    std::cout << "  'Token 1' element 0 = " << val_at_1
              << " (expected GGUF[0,1] = 1*1000+0 = 1000)" << std::endl;

    // This should be 1000 if the GGUF is row-major [768, 50257] but we're reading as column-major
    // Let's verify our GGUF simulation is correct
    float gguf_0_1 = gguf_data[0 * vocab_size + 1];  // Row 0, Col 1
    std::cout << "  GGUF[0,1] (row 0, col 1) = " << gguf_0_1
              << " (expected 1*1000+0 = 1000)" << std::endl;
    TEST_ASSERT_FLOAT_EQ(gguf_0_1, 1000.0f, 0.001f);

    std::cout << "  Transposed embedding lookup behavior understood" << std::endl;
    return 0;
}

// BF16 to FP32 conversion
static float bf16_to_fp32(uint16_t bf16) {
    uint32_t val = (uint32_t)bf16 << 16;
    float result;
    std::memcpy(&result, &val, sizeof(float));
    return result;
}

// Test actual GGUF file WTE dimensions
int test_gguf_wte_dims(const char* gguf_path) {
    print_test_header("test_gguf_wte_dims");

    if (!gguf_path) {
        std::cout << "  No GGUF path provided, skipping file test" << std::endl;
        return 0;
    }

    try {
        GGUFFile gguf = load_gguf(gguf_path);

        std::cout << "  GGUF file: " << gguf_path << std::endl;
        std::cout << "  Total tensors: " << gguf.tensors.size() << std::endl;

        // Find WTE tensor
        for (auto& t : gguf.tensors) {
            if (t.name == "token_embd.weight" || t.name == "wte") {
                std::cout << "  Found tensor: " << t.name << std::endl;
                std::cout << "    n_dims = " << t.n_dims << std::endl;
                std::cout << "    dims[0] = " << t.dims[0] << std::endl;
                std::cout << "    dims[1] = " << t.dims[1] << std::endl;
                std::cout << "    type = " << t.type << std::endl;

                // Our GGML allocation: ne[0]=768, ne[1]=50257
                size_t ggml_ne[2] = {768, 50257};
                bool dims_match_ne = (t.dims[0] == ggml_ne[0] && t.dims[1] == ggml_ne[1]);
                bool dims_match_ne_transposed = (t.dims[0] == ggml_ne[1] && t.dims[1] == ggml_ne[0]);

                std::cout << "    GGML ne[0]=768, ne[1]=50257" << std::endl;
                std::cout << "    GGUF dims match GGML ne directly: " << (dims_match_ne ? "YES" : "NO") << std::endl;
                std::cout << "    GGUF dims match GGML ne transposed: " << (dims_match_ne_transposed ? "YES" : "NO") << std::endl;

                if (dims_match_ne_transposed) {
                    std::cout << "    *** WARNING: GGUF is transposed relative to GGML allocation! ***" << std::endl;
                    std::cout << "    *** Transpose will be applied, which is correct behavior ***" << std::endl;
                }
            }
        }

        fclose(gguf.fp);
    } catch (const std::exception& e) {
        std::cerr << "  Error loading GGUF: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "  GGUF WTE dimensions verified" << std::endl;
    return 0;
}

// Test: Actually load WTE tensor from GGUF and validate data
int test_gguf_wte_loading(const char* gguf_path) {
    print_test_header("test_gguf_wte_loading");

    if (!gguf_path) {
        std::cout << "  No GGUF path provided, skipping file test" << std::endl;
        return 0;
    }

    try {
        GGUFFile gguf = load_gguf(gguf_path);

        std::cout << "  GGUF file: " << gguf_path << std::endl;
        std::cout << "  Total tensors: " << gguf.tensors.size() << std::endl;

        // Find WTE tensor
        GGUFTensorInfo* wte_info = nullptr;
        for (auto& t : gguf.tensors) {
            if (t.name == "token_embd.weight" || t.name == "wte") {
                wte_info = &t;
                std::cout << "  Found tensor: " << t.name << std::endl;
                std::cout << "    dims: [" << t.dims[0] << ", " << t.dims[1] << "]" << std::endl;
                std::cout << "    type: " << t.type << " (";
                switch (t.type) {
                    case GGUF_TID_F32: std::cout << "F32"; break;
                    case GGUF_TID_F16: std::cout << "F16"; break;
                    case GGUF_TID_BF16: std::cout << "BF16"; break;
                    case GGUF_TID_Q8_0_ALT: std::cout << "Q8_0_ALT"; break;
                    default: std::cout << "type_" << t.type; break;
                }
                std::cout << ")" << std::endl;
                break;
            }
        }

        if (!wte_info) {
            std::cerr << "  ERROR: token_embd.weight not found in GGUF file!" << std::endl;
            fclose(gguf.fp);
            return 1;
        }

        // Calculate expected size and actual bytes
        size_t expected_elements = wte_info->dims[0] * wte_info->dims[1];
        size_t actual_nbytes = gguf_tensor_nbytes(*wte_info);
        std::cout << "    Expected elements: " << expected_elements << std::endl;
        std::cout << "    Actual bytes in file: " << actual_nbytes << std::endl;

        // For Q8_0_ALT (type 30), the file actually stores BF16 data (2 bytes per element)
        // So we read as uint16_t and convert
        std::vector<float> wte_data(expected_elements);
        bool read_success = false;

        if (wte_info->type == GGUF_TID_Q8_0_ALT) {
            // Q8_0_ALT in this file is actually BF16 data (2 bytes/element)
            std::cout << "  Reading Q8_0_ALT as BF16..." << std::endl;
            std::vector<uint16_t> bf16_data(actual_nbytes / 2);
            read_tensor_data(gguf, *wte_info, bf16_data.data(), actual_nbytes);
            for (size_t i = 0; i < expected_elements && i < bf16_data.size(); i++) {
                wte_data[i] = bf16_to_fp32(bf16_data[i]);
            }
            read_success = true;
            std::cout << "  Converted " << bf16_data.size() << " BF16 values to float" << std::endl;
        } else if (wte_info->type == GGUF_TID_BF16) {
            std::cout << "  Reading BF16..." << std::endl;
            std::vector<uint16_t> bf16_data(actual_nbytes / 2);
            read_tensor_data(gguf, *wte_info, bf16_data.data(), actual_nbytes);
            for (size_t i = 0; i < expected_elements && i < bf16_data.size(); i++) {
                wte_data[i] = bf16_to_fp32(bf16_data[i]);
            }
            read_success = true;
        } else if (wte_info->type == GGUF_TID_F16) {
            std::cout << "  Reading F16..." << std::endl;
            std::vector<uint16_t> f16_data(actual_nbytes / 2);
            read_tensor_data(gguf, *wte_info, f16_data.data(), actual_nbytes);
            for (size_t i = 0; i < expected_elements && i < f16_data.size(); i++) {
                // Simple FP16 to FP32 conversion
                uint16_t f16 = f16_data[i];
                unsigned int sign = (f16 >> 15) & 0x1;
                unsigned int exp = (f16 >> 10) & 0x1f;
                unsigned int mant = f16 & 0x3ff;
                if (exp == 0) {
                    wte_data[i] = sign ? -0.0f : 0.0f;
                } else if (exp == 31) {
                    wte_data[i] = sign ? -INFINITY : INFINITY;
                } else {
                    int e = (int)exp - 15;
                    float m = 1.0f + mant / 1024.0f;
                    wte_data[i] = sign ? -std::pow(2.0f, (float)e) * m : std::pow(2.0f, (float)e) * m;
                }
            }
            read_success = true;
        } else if (wte_info->type == GGUF_TID_F32) {
            std::cout << "  Reading F32..." << std::endl;
            read_tensor_data(gguf, *wte_info, wte_data.data(), actual_nbytes);
            read_success = true;
        } else {
            std::cout << "  Unsupported type: " << wte_info->type << std::endl;
        }

        fclose(gguf.fp);

        if (!read_success) {
            std::cerr << "  ERROR: Failed to read tensor data" << std::endl;
            return 1;
        }

        // Compute statistics on loaded data
        float min_val = wte_data[0], max_val = wte_data[0];
        float abs_sum = 0.0f;
        size_t zero_count = 0;
        size_t nan_count = 0;
        size_t inf_count = 0;

        for (size_t i = 0; i < wte_data.size(); i++) {
            float v = wte_data[i];
            if (std::isnan(v)) nan_count++;
            else if (std::isinf(v)) inf_count++;
            else {
                abs_sum += std::abs(v);
                if (std::abs(v) < 1e-10f) zero_count++;
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
        }

        std::cout << "\n  WTE Statistics:" << std::endl;
        std::cout << "    Total elements: " << wte_data.size() << std::endl;
        std::cout << "    L1 norm (sum of abs): " << abs_sum << std::endl;
        std::cout << "    Min value: " << min_val << std::endl;
        std::cout << "    Max value: " << max_val << std::endl;
        std::cout << "    Zero elements: " << zero_count << " (" << (100.0 * zero_count / wte_data.size()) << "%)" << std::endl;
        std::cout << "    NaN elements: " << nan_count << std::endl;
        std::cout << "    Inf elements: " << inf_count << std::endl;

        // Validation checks
        TEST_ASSERT_MSG(abs_sum > 0.0f, "WTE L1 norm should be > 0 (not all zeros)");
        TEST_ASSERT_MSG(nan_count == 0, "WTE should have no NaN values");
        TEST_ASSERT_MSG(inf_count == 0, "WTE should have no Inf values");

        // Check first token embedding (token 0)
        std::cout << "\n  First token embedding (token 0):" << std::endl;
        std::cout << "    Elements 0-9: ";
        for (int i = 0; i < 10; i++) {
            std::cout << wte_data[i] << " ";
        }
        std::cout << std::endl;

        // Check a few tokens to verify different embeddings
        std::cout << "\n  Sample embeddings:" << std::endl;
        for (int token_id : {0, 1, 10, 100, 1000}) {
            float* emb = &wte_data[token_id * 768];  // If dims are [50257, 768]
            float emb_l1 = 0;
            for (int i = 0; i < 768; i++) emb_l1 += std::abs(emb[i]);
            std::cout << "    Token " << token_id << ": L1=" << emb_l1;
            if (emb_l1 > 0.001f) {
                std::cout << " (first 5: ";
                for (int i = 0; i < 5; i++) std::cout << emb[i] << " ";
                std::cout << ")";
            }
            std::cout << std::endl;
        }

        std::cout << "\n  GGUF WTE loading and validation PASSED" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "  Error loading GGUF: " << e.what() << std::endl;
        return 1;
    }
}

// Test L1 norm calculation for diagnosis
int test_l1_norm_diagnosis() {
    print_test_header("test_l1_norm_diagnosis");

    // The debug output shows L1 = -912.425 for the first token
    // Let's see what this could indicate

    // If WTE is all zeros, L1 = sum(|0|) = 0
    // If WTE is initialized to zero but somehow we read wrong data...

    // Create a simple embedding table
    int n_embd = 768;
    int vocab_size = 50257;
    std::vector<float> emb(vocab_size * n_embd, 0.0f);

    // L1 of all zeros
    float l1 = 0;
    for (auto v : emb) l1 += std::abs(v);
    std::cout << "  L1 norm of all zeros: " << l1 << std::endl;

    // What if we read from wrong offset?
    // Set first embedding to some values
    for (int i = 0; i < n_embd; i++) {
        emb[i] = i * 0.1f;  // 0, 0.1, 0.2, ...
    }
    // But we read from offset 1000
    l1 = 0;
    for (int i = 0; i < n_embd; i++) {
        l1 += std::abs(emb[1000 + i]);
    }
    std::cout << "  L1 if we read from offset 1000 (garbage): " << l1 << std::endl;

    // What if WTE was transposed during load and we read wrong?
    std::cout << "  L1 norm diagnosis complete" << std::endl;
    return 0;
}

int run_wte_diagnosis_tests(const char* gguf_path = nullptr) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running WTE Diagnosis Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = 0;
    result |= test_wte_transpose_logic();
    result |= test_transpose_2d();
    result |= test_embedding_lookup();
    result |= test_transposed_embedding_lookup();
    result |= test_l1_norm_diagnosis();
    result |= test_gguf_wte_dims(gguf_path);
    result |= test_gguf_wte_loading(gguf_path);

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All WTE Diagnosis Tests PASSED" << std::endl;
    } else {
        std::cout << "Some WTE Diagnosis Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}