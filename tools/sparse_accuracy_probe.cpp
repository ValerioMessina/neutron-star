#include "neutron/config.hpp"
#include "neutron/engine.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return text;
}

void filler(std::string& text, int count) {
    for (int i = 0; i < count; ++i) {
        text += " Archive entry " + std::to_string(i) +
                " describes an ordinary blue container beside a quiet stone path.";
    }
}
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: neutron-sparse-accuracy-probe MODEL dense|global|swa\n";
        return 2;
    }

    const std::string mode = argv[2];
    const bool global_sparse = mode == "global" || mode == "sparse";
    const bool swa_sparse = mode == "swa";
    if (mode != "dense" && !global_sparse && !swa_sparse) return 2;

    neutron::Config cfg;
    cfg.kv_cache = false;
    cfg.model_path = argv[1];
    if (const char * path = std::getenv("NEUTRON_METAL_FFN")) cfg.metal_ffn_path = path;
    cfg.batch = 768;
    cfg.context = 8192;
    cfg.sparse_context = global_sparse;
    cfg.sparse_context_threshold = 1024;
    cfg.sparse_context_sink = 256;
    cfg.sparse_context_window = 512;
    cfg.sparse_context_stride = 16;
    cfg.sparse_swa = swa_sparse;
    cfg.sparse_swa_threshold = 128;
    cfg.sparse_swa_sink = 64;
    cfg.sparse_swa_recent = 64;
    cfg.sparse_swa_stride = std::getenv("NEUTRON_SPARSE_SWA_STRIDE")
        ? static_cast<uint32_t>(std::stoul(std::getenv("NEUTRON_SPARSE_SWA_STRIDE")))
        : 2;

    auto engine = neutron::make_native_engine(cfg);
    std::string base =
        "<|turn>user\nMemorize the five labeled codes in this archive. "
        "The BEGIN code is ORCHID.";
    filler(base, 42);
    base += " The QUARTER code is LANTERN.";
    filler(base, 42);
    base += " The MIDDLE code is QUARTZ.";
    filler(base, 42);
    base += " The LATE code is SAFFRON.";
    filler(base, 42);
    base += " The RECENT code is COBALT.";
    filler(base, 8);

    const std::vector<std::pair<std::string, std::string>> cases = {
        {"BEGIN", "ORCHID"},
        {"QUARTER", "LANTERN"},
        {"MIDDLE", "QUARTZ"},
        {"LATE", "SAFFRON"},
        {"RECENT", "COBALT"},
    };
    neutron::SamplingParams sampling;
    sampling.max_tokens = 12;
    sampling.temperature = 0;

    int correct = 0;
    for (const auto& [label, expected] : cases) {
        const std::string prompt =
            base + " What is the " + label +
            " code? Reply with only that single code.<turn|>\n<|turn>model\n";
        const auto result = engine->generate(prompt, sampling);
        const bool ok = upper(result.text).find(expected) != std::string::npos;
        correct += ok;
        std::cout << "mode=" << argv[2] << " label=" << label
                  << " expected=" << expected << " correct=" << (ok ? 1 : 0)
                  << " prompt_tokens=" << result.stats.prompt_tokens
                  << " cached_tokens=" << result.stats.cached_tokens
                  << " answer=" << result.text << '\n';
    }
    std::cout << "mode=" << argv[2] << " correct=" << correct
              << " total=" << cases.size()
              << " accuracy=" << (100.0 * correct / cases.size()) << '\n';
}
