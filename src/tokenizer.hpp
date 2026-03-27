#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <regex>

// Custom hash for pair<int,int> since std::hash doesn't specialize for pairs
struct pair_hash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
    }
};

// GPT-2 Byte-Level BPE Tokenizer
// Pipeline: Raw String → Regex Chunks → UTF-8 Bytes → BPE Merges → Integer IDs

class GPT2Tokenizer {
public:
    GPT2Tokenizer();
    ~GPT2Tokenizer();

    // Load vocabulary and merges from files
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Encode string to token IDs
    std::vector<int> encode(const std::string& text);

    // Decode token IDs to string
    std::string decode(const std::vector<int>& tokens);

    // Special tokens
    static constexpr int EOS_TOKEN = 50256;
    static constexpr int PAD_TOKEN = 50257;  // Not used in GPT-2

private:
    // Unicode code point to UTF-8 bytes
    static void unicode_to_utf8(int codepoint, unsigned char* out);

    // Convert string to byte-level representation
    std::vector<unsigned char> string_to_bytes(const std::string& text);

    // Apply BPE merges to a sequence of bytes
    std::vector<int> apply_bpe(const std::vector<unsigned char>& bytes);

    // Byte pair encoding data
    std::unordered_map<int, int> byte_to_id_;       // byte -> token id
    std::unordered_map<int, std::string> id_to_token_; // token id -> decoded string
    std::unordered_map<std::pair<int, int>, int, pair_hash> merges_;  // (b1, b2) -> new_id
    std::vector<std::pair<int, int>> merge_order_;  // Ordered list of merges

    // Regex for GPT-2 tokenization
    std::regex pat_;
};

// Read file contents
std::string read_file(const std::string& path);

// Parse GPT-2 vocab file (json format with byte encoder)
std::unordered_map<int, std::string> parse_vocab(const std::string& path);

// Parse GPT-2 merges file
std::vector<std::pair<std::string, std::string>> parse_merges(const std::string& path);
