#include "neutron/config.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

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

bool parse_bool(std::string_view value, const char * option) {
    if (value.empty() || value == "0" || value == "false" || value == "off") return false;
    if (value == "1" || value == "true" || value == "on") return true;
    throw std::runtime_error(std::string("invalid boolean for ") + option + ": " + std::string(value));
}

uint32_t parse_uint(std::string_view value, const char * option) {
    try {
        size_t used = 0;
        const unsigned long long result = std::stoull(std::string(value), &used);
        if (used != value.size() || (!value.empty() && value.front() == '-') ||
            result > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("range");
        return static_cast<uint32_t>(result);
    } catch (...) {
        throw std::runtime_error(std::string("invalid unsigned integer for ") + option + ": " + std::string(value));
    }
}
} // namespace

Config Config::from_args(int argc, char ** argv) {
    Config c;
    c.model_path = env("NEUTRON_MODEL");
    c.draft_model_path = env("NEUTRON_DRAFT_MODEL");
    c.mmproj_path = env("NEUTRON_MMPROJ");
    c.metal_ffn_path = env("NEUTRON_METAL_FFN");
    c.kv_cache_dir = env("NEUTRON_KV_CACHE_DIR");
    c.api_key = env("NEUTRON_API_KEY");
    const std::string kv_cache = env("NEUTRON_KV_CACHE");
    if (!kv_cache.empty()) c.kv_cache = parse_bool(kv_cache, "NEUTRON_KV_CACHE");
    const std::string kv_cache_entries = env("NEUTRON_KV_CACHE_ENTRIES");
    if (!kv_cache_entries.empty()) c.kv_cache_entries = parse_uint(kv_cache_entries, "NEUTRON_KV_CACHE_ENTRIES");
    const std::string kv_cache_min_tokens = env("NEUTRON_KV_CACHE_MIN_TOKENS");
    if (!kv_cache_min_tokens.empty()) c.kv_cache_min_tokens = parse_uint(kv_cache_min_tokens, "NEUTRON_KV_CACHE_MIN_TOKENS");
    const std::string sparse_context = env("NEUTRON_SPARSE_CONTEXT");
    if (!sparse_context.empty()) c.sparse_context = parse_bool(sparse_context, "NEUTRON_SPARSE_CONTEXT");
    const std::string sparse_threshold = env("NEUTRON_SPARSE_THRESHOLD");
    if (!sparse_threshold.empty()) c.sparse_context_threshold = parse_uint(sparse_threshold, "NEUTRON_SPARSE_THRESHOLD");
    const std::string sparse_sink = env("NEUTRON_SPARSE_SINK");
    if (!sparse_sink.empty()) c.sparse_context_sink = parse_uint(sparse_sink, "NEUTRON_SPARSE_SINK");
    const std::string sparse_window = env("NEUTRON_SPARSE_WINDOW");
    if (!sparse_window.empty()) c.sparse_context_window = parse_uint(sparse_window, "NEUTRON_SPARSE_WINDOW");
    const std::string sparse_stride = env("NEUTRON_SPARSE_STRIDE");
    if (!sparse_stride.empty()) c.sparse_context_stride = parse_uint(sparse_stride, "NEUTRON_SPARSE_STRIDE");
    const std::string sparse_swa = env("NEUTRON_SPARSE_SWA");
    if (!sparse_swa.empty()) c.sparse_swa = parse_bool(sparse_swa, "NEUTRON_SPARSE_SWA");
    const std::string sparse_swa_threshold = env("NEUTRON_SPARSE_SWA_THRESHOLD");
    if (!sparse_swa_threshold.empty()) c.sparse_swa_threshold = parse_uint(sparse_swa_threshold, "NEUTRON_SPARSE_SWA_THRESHOLD");
    const std::string sparse_swa_sink = env("NEUTRON_SPARSE_SWA_SINK");
    if (!sparse_swa_sink.empty()) c.sparse_swa_sink = parse_uint(sparse_swa_sink, "NEUTRON_SPARSE_SWA_SINK");
    const std::string sparse_swa_recent = env("NEUTRON_SPARSE_SWA_RECENT");
    if (!sparse_swa_recent.empty()) c.sparse_swa_recent = parse_uint(sparse_swa_recent, "NEUTRON_SPARSE_SWA_RECENT");
    const std::string sparse_swa_stride = env("NEUTRON_SPARSE_SWA_STRIDE");
    if (!sparse_swa_stride.empty()) c.sparse_swa_stride = parse_uint(sparse_swa_stride, "NEUTRON_SPARSE_SWA_STRIDE");
    const std::string exact_ffn = env("NEUTRON_EXACT_FFN");
    if (!exact_ffn.empty()) c.exact_ffn = parse_bool(exact_ffn, "NEUTRON_EXACT_FFN");
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + a);
            return argv[i];
        };
        if (a == "-m" || a == "--model") c.model_path = next();
        else if (a == "--draft-model") c.draft_model_path = next();
        else if (a == "--mmproj") c.mmproj_path = next();
        else if (a == "--metal-ffn") c.metal_ffn_path = next();
        else if (a == "--kv-cache-dir") c.kv_cache_dir = next();
        else if (a == "--no-kv-cache") c.kv_cache = false;
        else if (a == "--kv-cache-entries") c.kv_cache_entries = parse_uint(next(), "--kv-cache-entries");
        else if (a == "--kv-cache-min-tokens") c.kv_cache_min_tokens = parse_uint(next(), "--kv-cache-min-tokens");
        else if (a == "--speculative-tokens") c.speculative_tokens = parse_int(next(), "--speculative-tokens");
        else if (a == "--model-name") c.model_name = next();
        else if (a == "--host") c.host = next();
        else if (a == "--port") c.port = parse_int(next(), "--port");
        else if (a == "-c" || a == "--ctx") c.context = parse_int(next(), "--ctx");
        else if (a == "-b" || a == "--batch") c.batch = parse_int(next(), "--batch");
        else if (a == "--sparse-context") c.sparse_context = true;
        else if (a == "--sparse-threshold") c.sparse_context_threshold = parse_uint(next(), "--sparse-threshold");
        else if (a == "--sparse-sink") c.sparse_context_sink = parse_uint(next(), "--sparse-sink");
        else if (a == "--sparse-window") c.sparse_context_window = parse_uint(next(), "--sparse-window");
        else if (a == "--sparse-stride") c.sparse_context_stride = parse_uint(next(), "--sparse-stride");
        else if (a == "--sparse-swa") c.sparse_swa = true;
        else if (a == "--dense-swa") c.sparse_swa = false;
        else if (a == "--sparse-swa-threshold") c.sparse_swa_threshold = parse_uint(next(), "--sparse-swa-threshold");
        else if (a == "--sparse-swa-sink") c.sparse_swa_sink = parse_uint(next(), "--sparse-swa-sink");
        else if (a == "--sparse-swa-recent") c.sparse_swa_recent = parse_uint(next(), "--sparse-swa-recent");
        else if (a == "--sparse-swa-stride") c.sparse_swa_stride = parse_uint(next(), "--sparse-swa-stride");
        else if (a == "--exact-ffn") c.exact_ffn = true;
        else if (a == "--api-key") c.api_key = next();
        else if (a == "-h" || a == "--help") throw std::runtime_error("help");
        else throw std::runtime_error("unknown option: " + a);
    }
    if (c.model_path.empty()) {
        c.model_path = env("HOME") + "/.ollama/models/blobs/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606";
    }
    if (c.port < 1 || c.port > 65535) throw std::runtime_error("port must be 1..65535");
    if (c.speculative_tokens < 1 || c.speculative_tokens > 3) throw std::runtime_error("speculative tokens must be 1..3");
    if (c.kv_cache_entries > 1024) throw std::runtime_error("KV cache entries must be 0..1024");
    if (c.context < 512 || c.batch < 1 || c.batch > 2048) throw std::runtime_error("context must be >= 512 and batch must be 1..2048");
    if (c.sparse_context_threshold < 1024) throw std::runtime_error("sparse threshold must be >= 1024");
    if (c.sparse_context_sink < 128) throw std::runtime_error("sparse sink must be >= 128");
    if (c.sparse_context_window < 128) throw std::runtime_error("sparse window must be >= 128");
    if (c.sparse_context_stride < 2 || c.sparse_context_stride > 128) throw std::runtime_error("sparse stride must be 2..128");
    if (c.sparse_swa_threshold < 64) throw std::runtime_error("SWA sparse threshold must be >= 64");
    if (c.sparse_swa_sink < 64) throw std::runtime_error("SWA sparse sink must be >= 64");
    if (c.sparse_swa_recent < 64) throw std::runtime_error("SWA sparse recent span must be >= 64");
    if (c.sparse_swa_stride < 2 || c.sparse_swa_stride > 128) throw std::runtime_error("SWA sparse stride must be 2..128");
    c.batch = std::min(c.batch, c.context);
    return c;
}

std::string Config::usage(const char * a) {
    return std::string("Usage: ") + a + " [options]\n"
        "  -m, --model PATH       Gemma 4 GGUF (defaults to the detected Ollama 12B blob)\n"
        "  -c, --ctx N            KV context tokens (default 8192)\n"
        "      --draft-model PATH Gemma 4 Assistant MTP GGUF\n"
        "      --mmproj PATH      Gemma 4 vision/audio projector GGUF\n"
        "      --metal-ffn PATH   preconverted K-major Metal FFN sidecar\n"
        "      --kv-cache-dir DIR indexed persistent Metal KV snapshots\n"
        "      --no-kv-cache      disable the default persistent KV cache\n"
        "      --kv-cache-entries N  maximum disk snapshots (default 8; 0 unlimited)\n"
        "      --kv-cache-min-tokens N  minimum snapshot length (default 256)\n"
        "      --speculative-tokens N  draft length, 1..3 (default 3)\n"
        "  -b, --batch N          quantized Metal prefill chunk, 1..2048 (default 1024)\n"
        "      --sparse-context   opt in to approximate sparse global attention\n"
        "      --sparse-threshold N  start sparsity at N tokens (default 65536)\n"
        "      --sparse-sink N    always-visible prefix tokens (default 1024)\n"
        "      --sparse-window N  always-visible recent tokens (default 32768)\n"
        "      --sparse-stride N  keep one old 128-token block every N (default 8)\n"
        "      --sparse-swa       use sparse attention in 40 sliding layers (default)\n"
        "      --dense-swa        restore exact attention in all sliding layers\n"
        "      --sparse-swa-threshold N  start SWA sparsity at N tokens (default 128)\n"
        "      --sparse-swa-sink N  dense oldest SWA tokens (default 64)\n"
        "      --sparse-swa-recent N  dense recent SWA tokens (default 64)\n"
        "      --sparse-swa-stride N  sample middle 64-token blocks (default 2)\n"
        "      --exact-ffn        use FP32 accumulation in all 48 FFN layers\n"
        "      --host ADDR        listen address (default 127.0.0.1)\n"
        "      --port N           listen port (default 8080)\n"
        "      --api-key KEY      require Bearer/x-api-key auth\n";
}

} // namespace neutron
