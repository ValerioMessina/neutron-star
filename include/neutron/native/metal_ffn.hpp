#pragma once

#include "neutron/native/gguf.hpp"
#include "neutron/native/model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace neutron::native {

enum class MetalFfnKind : uint32_t { Gate = 0, Up = 1, Down = 2 };

struct MetalFfnHeader {
    char magic[8];
    uint32_t version;
    uint32_t row_tile;
    uint32_t entry_count;
    uint32_t header_bytes;
    uint64_t model_size;
    uint64_t model_fingerprint;
    uint64_t file_size;
    uint64_t reserved[3];
};

struct MetalFfnEntry {
    uint32_t layer;
    uint32_t kind;
    uint32_t type;
    uint32_t cols;
    uint32_t rows;
    uint32_t reserved;
    uint64_t source_offset;
    uint64_t offset;
    uint64_t bytes;
};

static_assert(sizeof(MetalFfnHeader) == 72);
static_assert(sizeof(MetalFfnEntry) == 48);


uint64_t metal_ffn_fingerprint(const GGUF & gguf);
void convert_metal_ffn(const std::string & output_path, const GGUF & gguf,
                       const Gemma4Model & model);
void convert_metal_ffn_in_place(const std::string & model_path,
                                const std::string & metadata_path,
                                const GGUF & gguf, const Gemma4Model & model);

class MetalFfnFile {
public:
    MetalFfnFile(const std::string & path, const GGUF & gguf, const Gemma4Model & model);
    ~MetalFfnFile();
    MetalFfnFile(const MetalFfnFile &) = delete;
    MetalFfnFile & operator=(const MetalFfnFile &) = delete;

    const std::byte * mapped_data() const { return map_; }
    size_t file_size() const { return size_; }
    uint64_t offset(const Tensor & tensor) const;
    bool reordered(const Tensor & tensor) const;

private:
    int fd_ = -1;
    std::byte * map_ = nullptr;
    size_t size_ = 0;
    std::unordered_map<uint64_t, MetalFfnEntry> entries_;
};


} // namespace neutron::native
