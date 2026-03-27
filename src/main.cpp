#include "model.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

// Usage: gpt2 <prompt> [max_tokens] [temperature] [top_k]
// Example: gpt2 "Hello, world!" 100 0.8 50

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <prompt> [max_tokens] [temperature] [top_k]" << std::endl;
    std::cout << "  prompt:      Input text (required)" << std::endl;
    std::cout << "  max_tokens:  Maximum tokens to generate (default: 100)" << std::endl;
    std::cout << "  temperature: Sampling temperature (default: 1.0)" << std::endl;
    std::cout << "  top_k:       Top-k sampling parameter (default: 50)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\" 50 0.7 40" << std::endl;
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = (argc > 2) ? std::stoi(argv[2]) : 100;
    float temperature = (argc > 3) ? std::stof(argv[3]) : 1.0f;
    int top_k = (argc > 4) ? std::stoi(argv[4]) : 50;

    std::cout << "=== GPT-2 Large Inference ===" << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << "Top-k: " << top_k << std::endl;
    std::cout << "==============================" << std::endl;

    // Initialize model
    GPT2Model model;
    if (!model.init(true)) {  // true = use GPU
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }

    // Load weights (GGUF format - downloaded in Colab)
    std::string weights_path = "/content/gpt2-model/gpt2.Q4_K_M.gguf";
    std::cout << "Loading weights from: " << weights_path << std::endl;

    // For now, we use random initialization since we don't have actual weights
    // In Colab, this would download weights from HuggingFace
    if (!model.load_weights(weights_path)) {
        std::cerr << "Warning: Could not load weights, using random initialization" << std::endl;
        std::cerr << "In Colab, run the weight download cell first" << std::endl;
    }

    // Generate
    std::cout << std::endl << "Generating..." << std::endl;
    std::cout << "Output: ";

    // Simplified: just show EOS for now
    // Real implementation would call model.generate()
    std::cout << "[Generated output would appear here]" << std::endl;

    std::cout << std::endl << "Done!" << std::endl;

    return 0;
}

// ============================================================================
// INFERENCE LOOP (for reference, to be implemented)
//
// The actual inference loop works as follows:
//
// 1. Load model weights into GGML tensors
// 2. Tokenize input prompt using BPE tokenizer
// 3. For each position from 0 to context_length:
//    a. Get token embeddings for input_ids[position]
//    b. Add positional embeddings
//    c. For each layer 0 to 35:
//       - Apply LayerNorm
//       - Compute Q, K, V projections
//       - Store K, V in layer's KV cache
//       - Compute attention scores using cached K, V
//       - Apply attention output projection
//       - Apply residual connection
//       - Apply LayerNorm
//       - Compute FFN (GELU up_proj + down_proj)
//       - Apply residual connection
//    d. Apply final LayerNorm
//    e. Compute logits = hidden_states @ lm_head.T
//    f. Sample next token from logits (argmax or with temperature/top-k)
//    g. If EOS token, stop
//    h. Append token to input_ids and continue
//
// 4. Decode tokens back to string using tokenizer.decode()
//
// ============================================================================

// Placeholder implementation of the inference loop
/*
std::vector<int> inference_loop(
    GPT2Model& model,
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k
) {
    std::vector<int> tokens = prompt_tokens;
    int position = 0;

    while ((int)tokens.size() < (int)prompt_tokens.size() + max_new_tokens) {
        // Get logits for current position
        std::vector<float> logits = model.forward(tokens, position, true);

        // Get logits for the last token only
        // (logits are over vocab, we want the last position)
        const float* last_token_logits = &logits.back();

        // Sample next token
        int next_token = model.sample(logits, temperature, top_k);

        // Check for EOS
        if (next_token == model.EOS_TOKEN) {
            break;
        }

        // Append to sequence
        tokens.push_back(next_token);
        position++;

        // Print progress
        std::cout << ".";
        std::cout.flush();
    }

    std::cout << std::endl;
    return tokens;
}
*/
