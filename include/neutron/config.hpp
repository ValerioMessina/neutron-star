#pragma once

#include <cstdint>
#include <string>

namespace neutron {

struct Config {
    std::string model_path;
    std::string draft_model_path;
    std::string mmproj_path;
    std::string metal_ffn_path;
    std::string kv_cache_dir;
    std::string model_name = "gemma-4-12b";
    std::string host = "127.0.0.1";
    std::string api_key;
    int port = 8080;
    uint32_t context = 8192;
    uint32_t batch = 1024;
    uint32_t speculative_tokens = 3;
    uint32_t kv_cache_entries = 8;
    uint32_t kv_cache_min_tokens = 256;
    bool kv_cache = true;
    bool sparse_context = false;
    uint32_t sparse_context_threshold = 65536;
    uint32_t sparse_context_sink = 1024;
    uint32_t sparse_context_window = 32768;
    uint32_t sparse_context_stride = 8;
    bool sparse_swa = true;
    uint32_t sparse_swa_threshold = 128;
    uint32_t sparse_swa_sink = 64;
    uint32_t sparse_swa_recent = 64;
    uint32_t sparse_swa_stride = 2;
    bool exact_ffn = false;

    static Config from_args(int argc, char ** argv);
    static std::string usage(const char * argv0);
};

} // namespace neutron
