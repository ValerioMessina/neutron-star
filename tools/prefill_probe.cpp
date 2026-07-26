#include "neutron/config.hpp"
#include "neutron/engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {
void sparse_from_env(neutron::Config& cfg) {
    if (const char * enabled = std::getenv("NEUTRON_SPARSE_CONTEXT")) {
        const std::string value(enabled);
        cfg.sparse_context = value != "0" && value != "false" && value != "off";
        if (const char * v = std::getenv("NEUTRON_SPARSE_THRESHOLD")) cfg.sparse_context_threshold = std::stoul(v);
        if (const char * v = std::getenv("NEUTRON_SPARSE_SINK")) cfg.sparse_context_sink = std::stoul(v);
        if (const char * v = std::getenv("NEUTRON_SPARSE_WINDOW")) cfg.sparse_context_window = std::stoul(v);
        if (const char * v = std::getenv("NEUTRON_SPARSE_STRIDE")) cfg.sparse_context_stride = std::stoul(v);
    }
    if (const char * enabled_swa = std::getenv("NEUTRON_SPARSE_SWA")) {
        const std::string value(enabled_swa);
        cfg.sparse_swa = value != "0" && value != "false" && value != "off";
    }
    if (const char * v = std::getenv("NEUTRON_SPARSE_SWA_THRESHOLD")) cfg.sparse_swa_threshold = std::stoul(v);
    if (const char * v = std::getenv("NEUTRON_SPARSE_SWA_SINK")) cfg.sparse_swa_sink = std::stoul(v);
    if (const char * v = std::getenv("NEUTRON_SPARSE_SWA_RECENT")) cfg.sparse_swa_recent = std::stoul(v);
    if (const char * v = std::getenv("NEUTRON_SPARSE_SWA_STRIDE")) cfg.sparse_swa_stride = std::stoul(v);
    if (const char * exact = std::getenv("NEUTRON_EXACT_FFN")) {
        const std::string value(exact);
        cfg.exact_ffn = value != "0" && value != "false" && value != "off";
    }
}
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) return 2;
    neutron::Config cfg;
    cfg.kv_cache = false;
    cfg.model_path = argv[1];
    if (const char * path = std::getenv("NEUTRON_METAL_FFN")) cfg.metal_ffn_path = path;
    if (const char * path = std::getenv("NEUTRON_KV_CACHE_DIR")) { cfg.kv_cache = true; cfg.kv_cache_dir = path; }
    if (const char * value = std::getenv("NEUTRON_KV_CACHE_ENTRIES")) cfg.kv_cache_entries = std::stoul(value);
    if (const char * value = std::getenv("NEUTRON_KV_CACHE_MIN_TOKENS")) cfg.kv_cache_min_tokens = std::stoul(value);
    if (const char * path = std::getenv("NEUTRON_PROBE_DRAFT")) cfg.draft_model_path = path;
    cfg.batch = static_cast<uint32_t>(std::stoul(argv[2]));
    cfg.context = std::getenv("NEUTRON_PROBE_CONTEXT")
        ? static_cast<uint32_t>(std::stoul(std::getenv("NEUTRON_PROBE_CONTEXT")))
        : 4096;
    sparse_from_env(cfg);
    auto engine = neutron::make_native_engine(cfg);
    neutron::SamplingParams sampling;
    sampling.max_tokens = std::getenv("NEUTRON_PROBE_GEN_TOKENS")
        ? std::stoi(std::getenv("NEUTRON_PROBE_GEN_TOKENS"))
        : 1;
    sampling.temperature = 0;
    if (!std::getenv("NEUTRON_PROBE_NO_WARMUP")) {
        std::string warm = "warmup ";
        for (int i = 0; i < 48; ++i) warm += "Metal inference warmup sentence. ";
        (void)engine->generate(warm, sampling);
    }
    const int repeats = std::stoi(argv[3]);
    std::vector<double> rates;
    const int trials = argc == 5 ? std::max(1, std::stoi(argv[4])) : 3;
    for (int trial = 0; trial < trials; ++trial) {
        std::string prompt = "trial-" + std::to_string(trial) + " ";
        for (int i = 0; i < repeats; ++i) prompt += "Quantized inference processes the prompt efficiently on Apple silicon. ";
        auto result = engine->generate(prompt, sampling);
        const auto computed = result.stats.prompt_tokens - result.stats.cached_tokens;
        const double rate = 1000.0 * computed / result.stats.prefill_ms;
        rates.push_back(rate);
        std::cout << "trial=" << trial << " tokens=" << result.stats.prompt_tokens << " prefill_ms=" << result.stats.prefill_ms << " rate=" << rate
                  << " cached=" << result.stats.cached_tokens
                  << " text_hash=" << std::hash<std::string>{}(result.text) << '\n';
        if (std::getenv("NEUTRON_PRINT_TEXT")) std::cout << "text=" << result.text << '\n';
    }
    std::sort(rates.begin(), rates.end());
    std::cout << "median=" << rates[rates.size() / 2] << '\n';
}
