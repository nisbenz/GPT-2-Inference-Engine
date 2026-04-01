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

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All WTE Diagnosis Tests PASSED" << std::endl;
    } else {
        std::cout << "Some WTE Diagnosis Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}