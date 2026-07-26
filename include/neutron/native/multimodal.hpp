#pragma once

#include "neutron/engine.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace neutron::native {

struct PromptChunk {
    std::vector<int32_t> tokens;
    std::vector<float> embeddings;
    bool non_causal = false;

    size_t size() const {
        return tokens.empty() ? embeddings.size() / 3840 : tokens.size();
    }
};

class MultimodalProcessor {
public:
    MultimodalProcessor(const std::string & model_path,
                        const std::string & mmproj_path);
    ~MultimodalProcessor();

    MultimodalProcessor(const MultimodalProcessor &) = delete;
    MultimodalProcessor & operator=(const MultimodalProcessor &) = delete;

    std::vector<PromptChunk> encode(const std::string & prompt,
                                    const std::vector<MediaInput> & media);
    bool supports(const std::string & type) const;
    const std::string & marker() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neutron::native
