#pragma once

#include "neutron/config.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace neutron {

struct SamplingParams {
    int max_tokens = 1024;
    float temperature = 1.0F;
    float top_p = 0.95F;
    float min_p = 0.0F;
    int top_k = 64;
    uint32_t seed = 0xFFFFFFFFU;
    float repeat_penalty = 1.0F;
    std::vector<std::string> stop;
};

struct GenerationStats {
    uint64_t prompt_tokens = 0;
    uint64_t cached_tokens = 0;
    uint64_t generated_tokens = 0;
    double prefill_ms = 0;
    double generation_ms = 0;
    std::string finish_reason = "stop";
};

struct GenerationResult {
    std::string text;
    GenerationStats stats;
};

using TokenCallback = std::function<bool(const std::string &)>;

class Engine {
public:
    virtual ~Engine() = default;
    virtual GenerationResult generate(const std::string & prompt,
                                      const SamplingParams & params,
                                      const TokenCallback & callback = {}) = 0;
    virtual std::string model_description() const = 0;
    virtual uint64_t model_size() const = 0;
    virtual uint32_t context_size() const = 0;
};

std::unique_ptr<Engine> make_llama_engine(const Config & config);

} // namespace neutron
