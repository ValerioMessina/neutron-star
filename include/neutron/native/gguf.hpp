#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace neutron::native {

enum class TensorType : uint32_t { F32 = 0, F16 = 1, Q4_K = 12, Q6_K = 14 };

using Meta = std::variant<std::monostate, uint64_t, int64_t, double, bool, std::string,
    std::vector<uint8_t>, std::vector<int8_t>, std::vector<uint16_t>, std::vector<int16_t>,
    std::vector<uint32_t>, std::vector<int32_t>, std::vector<uint64_t>, std::vector<int64_t>,
    std::vector<float>, std::vector<double>, std::vector<std::string>>;

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    TensorType type{};
    uint64_t offset = 0;
    uint64_t bytes = 0;
    const std::byte * data = nullptr;
};

class GGUF {
public:
    explicit GGUF(const std::string & path, bool prefetch = true);
    ~GGUF();
    GGUF(const GGUF &) = delete;
    GGUF & operator=(const GGUF &) = delete;

    const Tensor & tensor(const std::string & name) const;
    bool has_tensor(const std::string & name) const { return tensor_index_.contains(name); }
    const std::vector<Tensor> & tensors() const { return tensors_; }
    bool has(const std::string & key) const { return metadata_.contains(key); }

    template<class T> const T & meta(const std::string & key) const {
        const auto it = metadata_.find(key);
        if (it == metadata_.end()) throw std::runtime_error("missing GGUF metadata: " + key);
        return std::get<T>(it->second);
    }
    uint64_t meta_u64(const std::string & key) const;
    double meta_number(const std::string & key) const;
    size_t data_offset() const { return data_offset_; }
    size_t file_size() const { return size_; }
    const std::byte * mapped_data() const { return map_; }

private:
    int fd_ = -1;
    std::byte * map_ = nullptr;
    size_t size_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, Meta> metadata_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

size_t tensor_type_size(TensorType type);
size_t tensor_block_size(TensorType type);

} // namespace neutron::native
