#pragma once

#include <cstdint>
#include <string>

namespace neutron {

struct Config {
    std::string model_path;
    std::string model_name = "gemma-4-12b";
    std::string host = "127.0.0.1";
    std::string api_key;
    std::string kv_cache_dir;
    int port = 8080;
    uint32_t context = 8192;
    uint32_t batch = 2048;
    uint32_t ubatch = 512;
    int gpu_layers = -1;
    int threads = 0;
    bool flash_attention = true;
    bool mlock = false;
    bool verbose = false;

    static Config from_args(int argc, char ** argv);
    static std::string usage(const char * argv0);
};

} // namespace neutron
