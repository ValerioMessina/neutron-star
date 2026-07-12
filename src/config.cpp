#include "neutron/config.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace neutron {
namespace {
std::string env(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "";
}

int parse_int(std::string_view value, const char * option) {
    try {
        size_t used = 0;
        int result = std::stoi(std::string(value), &used);
        if (used != value.size()) throw std::invalid_argument("tail");
        return result;
    } catch (...) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + std::string(value));
    }
}
} // namespace

Config Config::from_args(int argc, char ** argv) {
    Config c;
    c.model_path = env("NEUTRON_MODEL");
    c.api_key = env("NEUTRON_API_KEY");
    c.kv_cache_dir = env("NEUTRON_KV_CACHE_DIR");
    c.threads = std::max(1U, std::thread::hardware_concurrency() / 2);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + a);
            return argv[i];
        };
        if (a == "-m" || a == "--model") c.model_path = next();
        else if (a == "--model-name") c.model_name = next();
        else if (a == "--host") c.host = next();
        else if (a == "--port") c.port = parse_int(next(), "--port");
        else if (a == "-c" || a == "--ctx") c.context = parse_int(next(), "--ctx");
        else if (a == "-b" || a == "--batch") c.batch = parse_int(next(), "--batch");
        else if (a == "--ubatch") c.ubatch = parse_int(next(), "--ubatch");
        else if (a == "--threads") c.threads = parse_int(next(), "--threads");
        else if (a == "--gpu-layers") c.gpu_layers = parse_int(next(), "--gpu-layers");
        else if (a == "--api-key") c.api_key = next();
        else if (a == "--kv-cache-dir") c.kv_cache_dir = next();
        else if (a == "--no-flash-attn") c.flash_attention = false;
        else if (a == "--mlock") c.mlock = true;
        else if (a == "-v" || a == "--verbose") c.verbose = true;
        else if (a == "-h" || a == "--help") throw std::runtime_error("help");
        else throw std::runtime_error("unknown option: " + a);
    }
    if (c.model_path.empty()) {
        c.model_path = env("HOME") + "/.ollama/models/blobs/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606";
    }
    if (c.port < 1 || c.port > 65535) throw std::runtime_error("port must be 1..65535");
    if (c.context < 512 || c.batch < 1 || c.ubatch < 1) throw std::runtime_error("invalid context/batch configuration");
    c.batch = std::min(c.batch, c.context);
    c.ubatch = std::min(c.ubatch, c.batch);
    return c;
}

std::string Config::usage(const char * a) {
    return std::string("Usage: ") + a + " [options]\n"
        "  -m, --model PATH       Gemma 4 GGUF (defaults to the detected Ollama 12B blob)\n"
        "  -c, --ctx N            KV context tokens (default 8192)\n"
        "  -b, --batch N          logical prefill batch (default 2048)\n"
        "      --ubatch N         physical Metal batch (default 512)\n"
        "      --host ADDR        listen address (default 127.0.0.1)\n"
        "      --port N           listen port (default 8080)\n"
        "      --api-key KEY      require Bearer/x-api-key auth\n"
        "      --kv-cache-dir DIR persist the active KV state\n"
        "      --no-flash-attn    disable flash attention\n"
        "      --gpu-layers N     -1 offloads all layers\n";
}

} // namespace neutron
