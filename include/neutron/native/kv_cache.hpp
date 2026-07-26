#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace neutron::native {

struct KvCacheLayer {
    uint32_t layer = 0;
    uint32_t capacity = 0;
    uint32_t dim = 0;
    uint32_t heads = 0;
    uint32_t slots = 0;
    uint32_t reserved = 0;
    uint64_t bytes_per_head = 0;
    uint64_t key_offset = 0;
    uint64_t value_offset = 0;
};

static_assert(sizeof(KvCacheLayer) == 48);

class KvCacheMapping {
public:
    ~KvCacheMapping();
    KvCacheMapping(const KvCacheMapping &) = delete;
    KvCacheMapping & operator=(const KvCacheMapping &) = delete;

    const std::byte * data() const noexcept { return map_; }
    size_t size() const noexcept { return size_; }
    size_t token_count() const noexcept { return tokens_.size(); }
    uint64_t payload_offset() const noexcept { return payload_offset_; }
    std::span<const int32_t> tokens() const noexcept { return tokens_; }
    std::span<const KvCacheLayer> layers() const noexcept { return layers_; }

private:
    friend class KvCacheIndex;
    KvCacheMapping() = default;

    int fd_ = -1;
    std::byte * map_ = nullptr;
    size_t size_ = 0;
    uint64_t payload_offset_ = 0;
    std::span<const int32_t> tokens_;
    std::span<const KvCacheLayer> layers_;
};

// Content-addressed, process-independent index for native Metal KV snapshots.
// Filenames contain the model fingerprint, token count and rolling token hash,
// so longest-prefix lookup does not need to scan multi-hundred-megabyte files.
class KvCacheIndex {
public:
    KvCacheIndex(std::string directory, uint64_t model_size,
                 uint64_t model_fingerprint, uint32_t context,
                 uint32_t batch, uint32_t max_entries);

    bool enabled() const noexcept { return !directory_.empty(); }
    std::unique_ptr<KvCacheMapping> find_longest(
        std::span<const int32_t> input, size_t longer_than) const;
    void store(std::span<const int32_t> tokens,
               std::span<const KvCacheLayer> layers,
               const void * payload, uint64_t payload_bytes);

    static uint64_t token_hash(std::span<const int32_t> tokens);

private:
    struct Candidate {
        uint32_t token_count = 0;
        uint64_t token_hash = 0;
        std::filesystem::path path;
    };

    std::unique_ptr<KvCacheMapping> open_candidate(
        const Candidate & candidate, std::span<const int32_t> input) const;
    void refresh();
    void prune();
    std::string file_prefix() const;

    std::string directory_;
    uint64_t model_size_ = 0;
    uint64_t model_fingerprint_ = 0;
    uint32_t context_ = 0;
    uint32_t batch_ = 0;
    uint32_t max_entries_ = 0;
    mutable std::mutex mutex_;
    std::unordered_multimap<uint64_t,Candidate> candidates_;
};

} // namespace neutron::native
