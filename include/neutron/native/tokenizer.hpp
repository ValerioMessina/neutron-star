#pragma once

#include "neutron/native/gguf.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace neutron::native {

class Gemma4Tokenizer {
public:
    explicit Gemma4Tokenizer(const GGUF & gguf);
    std::vector<int32_t> encode(const std::string & text, bool add_bos = true) const;
    std::string decode(int32_t token, bool render_special = false) const;
    std::string decode(const std::vector<int32_t> & tokens, bool render_special = false) const;
    int32_t token_id(const std::string & piece) const;
    size_t size() const { return tokens_.size(); }
    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    bool is_eog(int32_t token) const;

private:
    void encode_plain(std::string text, std::vector<int32_t> & out) const;
    void encode_run(const std::string & run, std::vector<int32_t> & out) const;

    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;
    std::unordered_map<std::string, int32_t> ids_;
    std::unordered_map<std::string, int32_t> ranks_;
    std::vector<std::pair<std::string, int32_t>> specials_;
    int32_t bos_ = 2;
    int32_t eos_ = 1;
};

} // namespace neutron::native
