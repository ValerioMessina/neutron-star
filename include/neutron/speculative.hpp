#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace neutron {

// Greedy EAGLE/MTP verification. The target pass supplies one prediction for
// every drafted position plus the bonus prediction after an entirely accepted
// draft. This preserves the target model's ordinary greedy output exactly.
struct MtpVerification {
    size_t accepted = 0;
    int32_t next_token = 0;
};

inline MtpVerification verify_mtp_greedy(std::span<const int32_t> draft,
                                         std::span<const int32_t> target_predictions) {
    if (target_predictions.size() != draft.size() + 1) {
        throw std::invalid_argument("MTP verification needs draft_size + 1 target predictions");
    }
    for (size_t i = 0; i < draft.size(); ++i) {
        if (draft[i] != target_predictions[i]) return {i, target_predictions[i]};
    }
    return {draft.size(), target_predictions.back()};
}

} // namespace neutron
