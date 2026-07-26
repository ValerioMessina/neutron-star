#include "neutron/native/metal_ffn.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace neutron::native {
namespace {
constexpr std::array<char,8> kMagic{'N','S','M','F','F','N','0','3'};
constexpr uint32_t kVersion = 3;
constexpr uint32_t kRowTile = 64;
constexpr uint32_t kHeaderBytes = 65536;

struct FfnTensorRef {
    uint32_t layer;
    uint32_t kind;
    const Tensor * tensor;
};

void hash_bytes(uint64_t & hash, const std::byte * data, size_t size) {
    constexpr uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint8_t>(data[i]);
        hash *= prime;
    }
}

void pwrite_all(int fd, const void * data, size_t size, uint64_t offset) {
    const auto * p = static_cast<const std::byte *>(data);
    while (size) {
        const ssize_t n = ::pwrite(fd, p, size, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("Metal FFN write failed: " + std::string(std::strerror(errno)));
        }
        p += n; size -= static_cast<size_t>(n); offset += static_cast<uint64_t>(n);
    }
}

void pread_all(int fd, void * data, size_t size, uint64_t offset) {
    auto * p = static_cast<std::byte *>(data);
    while (size) {
        const ssize_t n = ::pread(fd, p, size, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("Metal FFN read failed: " + std::string(std::strerror(errno)));
        }
        if (n == 0) throw std::runtime_error("unexpected end of Metal FFN source");
        p += n; size -= static_cast<size_t>(n); offset += static_cast<uint64_t>(n);
    }
}

std::unordered_map<uint64_t,std::pair<uint32_t,uint32_t>> ffn_tensors(const Gemma4Model & model) {
    std::unordered_map<uint64_t,std::pair<uint32_t,uint32_t>> result;
    for (uint32_t i = 0; i < model.layer.size(); ++i) {
        result.emplace(model.layer[i].gate->offset, std::pair{i, uint32_t(MetalFfnKind::Gate)});
        result.emplace(model.layer[i].up->offset, std::pair{i, uint32_t(MetalFfnKind::Up)});
        result.emplace(model.layer[i].down->offset, std::pair{i, uint32_t(MetalFfnKind::Down)});
    }
    return result;
}

void validate_tensor(const Tensor & tensor) {
    if (tensor.shape.size() != 2 || tensor.shape[0] % 256 || tensor.shape[1] % kRowTile)
        throw std::runtime_error("Metal FFN tensor has unsupported shape: " + tensor.name);
    if (tensor.type != TensorType::Q4_K && tensor.type != TensorType::Q6_K)
        throw std::runtime_error("Metal FFN tensor has unsupported type: " + tensor.name);
}
} // namespace

uint64_t metal_ffn_fingerprint(const GGUF & gguf) {
    uint64_t hash = 1469598103934665603ULL;
    hash_bytes(hash, reinterpret_cast<const std::byte *>(&kVersion), sizeof(kVersion));
    const uint64_t size = gguf.file_size();
    hash_bytes(hash, reinterpret_cast<const std::byte *>(&size), sizeof(size));
    const auto * data = gguf.mapped_data();
    const size_t header = std::min<size_t>(gguf.data_offset(), 16U << 20);
    hash_bytes(hash, data, header);
    return hash;
}

void convert_metal_ffn(const std::string & output_path, const GGUF & gguf,
                       const Gemma4Model & model) {
    const auto & weights = gguf.tensors();
    const auto ffn = ffn_tensors(model);
    if (sizeof(MetalFfnHeader) + weights.size() * sizeof(MetalFfnEntry) > kHeaderBytes)
        throw std::runtime_error("Metal FFN header capacity exceeded");

    MetalFfnHeader header{};
    std::copy(kMagic.begin(), kMagic.end(), header.magic);
    header.version = kVersion;
    header.row_tile = kRowTile;
    header.entry_count = static_cast<uint32_t>(weights.size());
    header.header_bytes = kHeaderBytes;
    header.model_size = gguf.file_size();
    header.model_fingerprint = metal_ffn_fingerprint(gguf);

    std::vector<MetalFfnEntry> entries;
    entries.reserve(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        const Tensor & tensor = weights[i];
        const auto fi = ffn.find(tensor.offset);
        if (fi != ffn.end()) validate_tensor(tensor);
        entries.push_back({fi == ffn.end() ? UINT32_MAX : fi->second.first,
            fi == ffn.end() ? UINT32_MAX : fi->second.second,
            static_cast<uint32_t>(tensor.type), static_cast<uint32_t>(tensor.shape[0]),
            static_cast<uint32_t>(tensor.shape.size()>1?tensor.shape[1]:1), 0, tensor.offset,
            gguf.data_offset()+tensor.offset, tensor.bytes});
    }
    header.file_size = gguf.file_size();

    const std::filesystem::path final(output_path);
    if (!final.parent_path().empty()) std::filesystem::create_directories(final.parent_path());
    const std::string temporary = output_path + ".tmp";
    const int fd = ::open(temporary.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) throw std::runtime_error("cannot create Metal FFN file: " + temporary);
    try {
        if (::ftruncate(fd, static_cast<off_t>(header.file_size)) != 0)
            throw std::runtime_error("cannot size Metal FFN file: " + std::string(std::strerror(errno)));
        std::array<std::byte,kHeaderBytes> header_page{};
        std::memcpy(header_page.data(), &header, sizeof(header));
        std::memcpy(header_page.data() + sizeof(header), entries.data(), entries.size() * sizeof(MetalFfnEntry));
        pwrite_all(fd, header_page.data(), header_page.size(), 0);

        for (size_t i = 0; i < weights.size(); ++i) {
            const Tensor & tensor = weights[i];
            if (ffn.contains(tensor.offset)) {
                const uint64_t cols = tensor.shape[0], rows = tensor.shape[1];
                const uint64_t blocks = cols / 256;
                const size_t block_bytes = tensor_type_size(tensor.type);
                std::vector<std::byte> reordered(tensor.bytes);
                for (uint64_t row0 = 0; row0 < rows; row0 += kRowTile)
                    for (uint64_t block = 0; block < blocks; ++block)
                        for (uint64_t row = 0; row < kRowTile; ++row) {
                            const uint64_t source = ((row0 + row) * blocks + block) * block_bytes;
                            const uint64_t target = (((row0 / kRowTile) * blocks + block) * kRowTile + row) * block_bytes;
                            std::memcpy(reordered.data() + target, tensor.data + source, block_bytes);
                        }
                pwrite_all(fd, reordered.data(), reordered.size(), entries[i].offset);
            } else {
                pwrite_all(fd, tensor.data, tensor.bytes, entries[i].offset);
            }
            std::cerr << "metal-model " << (i + 1) << '/' << weights.size() << ' '
                      << tensor.name << "\n";
        }
        if (::fsync(fd) != 0) throw std::runtime_error("Metal FFN fsync failed");
        ::close(fd);
        if (::rename(temporary.c_str(), output_path.c_str()) != 0)
            throw std::runtime_error("Metal FFN rename failed: " + std::string(std::strerror(errno)));
    } catch (...) {
        ::close(fd);
        ::unlink(temporary.c_str());
        throw;
    }
}

void convert_metal_ffn_in_place(const std::string & model_path,
                                const std::string & metadata_path,
                                const GGUF & gguf, const Gemma4Model & model) {
    const auto & weights = gguf.tensors();
    const auto ffn = ffn_tensors(model);
    if (sizeof(MetalFfnHeader) + weights.size() * sizeof(MetalFfnEntry) > kHeaderBytes)
        throw std::runtime_error("Metal FFN header capacity exceeded");
    if (gguf.data_offset() < kHeaderBytes)
        throw std::runtime_error("GGUF metadata area is too small for the Metal header");

    MetalFfnHeader header{};
    std::copy(kMagic.begin(), kMagic.end(), header.magic);
    header.version = kVersion;
    header.row_tile = kRowTile;
    header.entry_count = static_cast<uint32_t>(weights.size());
    header.header_bytes = kHeaderBytes;
    header.model_size = gguf.file_size();
    header.model_fingerprint = metal_ffn_fingerprint(gguf);
    header.file_size = gguf.file_size();

    std::vector<MetalFfnEntry> entries;
    entries.reserve(weights.size());
    for (const Tensor & tensor : weights) {
        const auto fi = ffn.find(tensor.offset);
        if (fi != ffn.end()) validate_tensor(tensor);
        entries.push_back({fi == ffn.end() ? UINT32_MAX : fi->second.first,
            fi == ffn.end() ? UINT32_MAX : fi->second.second,
            static_cast<uint32_t>(tensor.type), static_cast<uint32_t>(tensor.shape[0]),
            static_cast<uint32_t>(tensor.shape.size()>1?tensor.shape[1]:1), 0, tensor.offset,
            gguf.data_offset()+tensor.offset, tensor.bytes});
    }

    const std::filesystem::path meta(metadata_path);
    if (!meta.parent_path().empty()) std::filesystem::create_directories(meta.parent_path());
    const std::string meta_tmp = metadata_path + ".tmp";
    int metadata_fd = ::open(meta_tmp.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (metadata_fd < 0) throw std::runtime_error("cannot create sparse GGUF metadata file");
    try {
        if (::ftruncate(metadata_fd, static_cast<off_t>(gguf.file_size())) != 0)
            throw std::runtime_error("cannot size sparse GGUF metadata file");
        pwrite_all(metadata_fd, gguf.mapped_data(), gguf.data_offset(), 0);
        if (::fsync(metadata_fd) != 0) throw std::runtime_error("GGUF metadata fsync failed");
        ::close(metadata_fd); metadata_fd = -1;
        if (::rename(meta_tmp.c_str(), metadata_path.c_str()) != 0)
            throw std::runtime_error("GGUF metadata rename failed: " + std::string(std::strerror(errno)));
    } catch (...) {
        if (metadata_fd >= 0) ::close(metadata_fd);
        ::unlink(meta_tmp.c_str());
        throw;
    }

    const int fd = ::open(model_path.c_str(), O_RDWR);
    if (fd < 0) throw std::runtime_error("cannot open model for in-place Metal conversion");
    try {
        for (size_t i = 0; i < weights.size(); ++i) {
            const Tensor & tensor = weights[i];
            if (!ffn.contains(tensor.offset)) continue;
            const uint64_t cols = tensor.shape[0], rows = tensor.shape[1];
            const uint64_t blocks = cols / 256;
            const size_t block_bytes = tensor_type_size(tensor.type);
            const size_t slab_bytes = kRowTile * blocks * block_bytes;
            std::vector<std::byte> source(slab_bytes), reordered(slab_bytes);
            for (uint64_t row0 = 0; row0 < rows; row0 += kRowTile) {
                const uint64_t slab_offset = entries[i].offset + row0 * blocks * block_bytes;
                pread_all(fd, source.data(), source.size(), slab_offset);
                for (uint64_t block = 0; block < blocks; ++block)
                    for (uint64_t row = 0; row < kRowTile; ++row)
                        std::memcpy(reordered.data() + (block*kRowTile+row)*block_bytes,
                                    source.data() + (row*blocks+block)*block_bytes, block_bytes);
                pwrite_all(fd, reordered.data(), reordered.size(), slab_offset);
            }
            std::cerr << "metal-in-place " << (i + 1) << '/' << weights.size() << ' '
                      << tensor.name << "\n";
        }
        std::array<std::byte,kHeaderBytes> header_page{};
        std::memcpy(header_page.data(), &header, sizeof(header));
        std::memcpy(header_page.data() + sizeof(header), entries.data(), entries.size() * sizeof(MetalFfnEntry));
        pwrite_all(fd, header_page.data(), header_page.size(), 0);
        if (::fsync(fd) != 0) throw std::runtime_error("in-place Metal model fsync failed");
        ::close(fd);
    } catch (...) {
        ::close(fd);
        throw;
    }
}

MetalFfnFile::MetalFfnFile(const std::string & path, const GGUF & gguf,
                           const Gemma4Model & model) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open Metal FFN file: " + path);
    try {
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(kHeaderBytes))
            throw std::runtime_error("invalid Metal FFN file: " + path);
        size_ = static_cast<size_t>(st.st_size);
        map_ = static_cast<std::byte *>(::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("mmap Metal FFN failed"); }
        const auto & header = *reinterpret_cast<const MetalFfnHeader *>(map_);
        if (!std::equal(kMagic.begin(), kMagic.end(), header.magic) || header.version != kVersion ||
            header.row_tile != kRowTile || header.header_bytes != kHeaderBytes ||
            header.entry_count != gguf.tensors().size() || header.file_size != size_ ||
            header.model_size != gguf.file_size() || header.model_fingerprint != metal_ffn_fingerprint(gguf) ||
            sizeof(header) + size_t(header.entry_count) * sizeof(MetalFfnEntry) > header.header_bytes)
            throw std::runtime_error("Metal FFN file does not match this GGUF");
        const auto * entry = reinterpret_cast<const MetalFfnEntry *>(map_ + sizeof(MetalFfnHeader));
        const auto & expected = gguf.tensors();
        const auto ffn = ffn_tensors(model);
        for (uint32_t i = 0; i < header.entry_count; ++i) {
            const Tensor & tensor = expected[i];
            const auto & e = entry[i];
            const auto fi = ffn.find(tensor.offset);
            const uint32_t layer = fi == ffn.end() ? UINT32_MAX : fi->second.first;
            const uint32_t kind = fi == ffn.end() ? UINT32_MAX : fi->second.second;
            if (e.layer != layer || e.kind != kind || e.type != static_cast<uint32_t>(tensor.type) ||
                e.cols != tensor.shape[0] || e.rows != (tensor.shape.size()>1?tensor.shape[1]:1) ||
                e.source_offset != tensor.offset || e.bytes != tensor.bytes ||
                e.offset > size_ || e.bytes > size_ - e.offset)
                throw std::runtime_error("invalid Metal FFN tensor entry " + std::to_string(i));
            if (!entries_.emplace(e.source_offset, e).second)
                throw std::runtime_error("duplicate Metal FFN tensor entry");
        }
        ::madvise(map_, size_, MADV_RANDOM);
    } catch (...) {
        if (map_) { ::munmap(map_, size_); map_ = nullptr; }
        ::close(fd_); fd_ = -1;
        throw;
    }
}

MetalFfnFile::~MetalFfnFile() {
    if (map_) ::munmap(map_, size_);
    if (fd_ >= 0) ::close(fd_);
}

uint64_t MetalFfnFile::offset(const Tensor & tensor) const {
    const auto it = entries_.find(tensor.offset);
    if (it == entries_.end()) throw std::runtime_error("tensor missing from Metal FFN file: " + tensor.name);
    return it->second.offset;
}

bool MetalFfnFile::reordered(const Tensor & tensor) const {
    const auto it = entries_.find(tensor.offset);
    if (it == entries_.end()) throw std::runtime_error("tensor missing from Metal FFN file: " + tensor.name);
    return it->second.kind != UINT32_MAX;
}

} // namespace neutron::native
