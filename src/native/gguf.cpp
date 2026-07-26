#include "neutron/native/gguf.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace neutron::native {
namespace {
enum class ValueType : uint32_t { U8, I8, U16, I16, U32, I32, F32, Bool, String, Array, U64, I64, F64 };

class Reader {
public:
    Reader(const std::byte * data, size_t size) : data_(data), size_(size) {}
    template<class T> T read() {
        if (pos_ + sizeof(T) > size_) throw std::runtime_error("truncated GGUF header");
        T value; std::memcpy(&value, data_ + pos_, sizeof(T)); pos_ += sizeof(T); return value;
    }
    std::string string() {
        const uint64_t n = read<uint64_t>();
        if (n > size_ - pos_) throw std::runtime_error("invalid GGUF string length");
        std::string out(reinterpret_cast<const char *>(data_ + pos_), static_cast<size_t>(n)); pos_ += n; return out;
    }
    size_t tell() const { return pos_; }
private:
    const std::byte * data_; size_t size_; size_t pos_ = 0;
};

template<class T> std::vector<T> read_array(Reader & r, uint64_t n) {
    if (n > (1ULL << 31)) throw std::runtime_error("unreasonable GGUF array length");
    std::vector<T> out; out.reserve(n);
    for (uint64_t i = 0; i < n; ++i) out.push_back(r.read<T>());
    return out;
}

Meta read_value(Reader & r, ValueType type, uint64_t n, bool array) {
    if (!array) {
        switch (type) {
            case ValueType::U8: return uint64_t(r.read<uint8_t>()); case ValueType::I8: return int64_t(r.read<int8_t>());
            case ValueType::U16: return uint64_t(r.read<uint16_t>()); case ValueType::I16: return int64_t(r.read<int16_t>());
            case ValueType::U32: return uint64_t(r.read<uint32_t>()); case ValueType::I32: return int64_t(r.read<int32_t>());
            case ValueType::U64: return r.read<uint64_t>(); case ValueType::I64: return r.read<int64_t>();
            case ValueType::F32: return double(r.read<float>()); case ValueType::F64: return r.read<double>();
            case ValueType::Bool: return r.read<uint8_t>() != 0; case ValueType::String: return r.string();
            default: throw std::runtime_error("unsupported GGUF scalar type");
        }
    }
    switch (type) {
        case ValueType::U8: return read_array<uint8_t>(r,n); case ValueType::I8: return read_array<int8_t>(r,n);
        case ValueType::U16: return read_array<uint16_t>(r,n); case ValueType::I16: return read_array<int16_t>(r,n);
        case ValueType::U32: return read_array<uint32_t>(r,n); case ValueType::I32: return read_array<int32_t>(r,n);
        case ValueType::U64: return read_array<uint64_t>(r,n); case ValueType::I64: return read_array<int64_t>(r,n);
        case ValueType::F32: return read_array<float>(r,n); case ValueType::F64: return read_array<double>(r,n);
        case ValueType::Bool: return read_array<uint8_t>(r,n);
        case ValueType::String: { std::vector<std::string> out; out.reserve(n); for (uint64_t i=0;i<n;++i) out.push_back(r.string()); return out; }
        default: throw std::runtime_error("nested/invalid GGUF array type");
    }
}
} // namespace

size_t tensor_type_size(TensorType t) {
    switch (t) { case TensorType::F32: return 4; case TensorType::F16: return 2; case TensorType::Q4_K: return 144; case TensorType::Q6_K: return 210; }
    throw std::runtime_error("unsupported tensor type " + std::to_string(static_cast<uint32_t>(t)));
}
size_t tensor_block_size(TensorType t) { return (t == TensorType::Q4_K || t == TensorType::Q6_K) ? 256 : 1; }

GGUF::GGUF(const std::string & path, bool prefetch) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open GGUF: " + path);
    struct stat st{};
    if (fstat(fd_, &st) != 0 || st.st_size < 24) { ::close(fd_); fd_=-1; throw std::runtime_error("invalid GGUF file"); }
    size_ = static_cast<size_t>(st.st_size);
    map_ = static_cast<std::byte *>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (map_ == MAP_FAILED) { map_=nullptr; ::close(fd_); fd_=-1; throw std::runtime_error("mmap GGUF failed"); }
    madvise(map_, size_, MADV_RANDOM);

    Reader r(map_, size_);
    if (r.read<uint32_t>() != 0x46554747U) throw std::runtime_error("bad GGUF magic");
    const uint32_t version = r.read<uint32_t>();
    if (version != 3) throw std::runtime_error("only GGUF v3 is supported");
    const uint64_t tensor_count = r.read<uint64_t>();
    const uint64_t kv_count = r.read<uint64_t>();
    if (tensor_count > 100000 || kv_count > 100000) throw std::runtime_error("invalid GGUF counts");
    for (uint64_t i=0;i<kv_count;++i) {
        auto key = r.string();
        auto type = static_cast<ValueType>(r.read<uint32_t>());
        bool array = type == ValueType::Array; uint64_t n = 1;
        if (array) { type = static_cast<ValueType>(r.read<uint32_t>()); n = r.read<uint64_t>(); }
        metadata_.emplace(std::move(key), read_value(r,type,n,array));
    }
    size_t alignment = 32;
    if (has("general.alignment")) alignment = meta_u64("general.alignment");
    tensors_.reserve(tensor_count);
    for (uint64_t i=0;i<tensor_count;++i) {
        Tensor t; t.name = r.string();
        const uint32_t nd = r.read<uint32_t>(); if (nd == 0 || nd > 4) throw std::runtime_error("invalid tensor rank");
        for (uint32_t d=0;d<nd;++d) t.shape.push_back(r.read<uint64_t>());
        t.type = static_cast<TensorType>(r.read<uint32_t>()); t.offset = r.read<uint64_t>();
        uint64_t elements=1; for (auto d:t.shape) elements*=d;
        try {
            const size_t block=tensor_block_size(t.type); if (elements%block) throw std::runtime_error("invalid quantized tensor shape");
            t.bytes=elements/block*tensor_type_size(t.type);
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string(e.what()) + ": " + t.name);
        }
        tensor_index_.emplace(t.name,tensors_.size()); tensors_.push_back(std::move(t));
    }
    data_offset_ = (r.tell()+alignment-1)&~(alignment-1);
    for (auto & t:tensors_) {
        if (data_offset_+t.offset+t.bytes>size_) throw std::runtime_error("tensor outside GGUF mapping: "+t.name);
        t.data=map_+data_offset_+t.offset;
    }
    // Metal reads the mmap directly.  Ask the VM to page the GGUF in ahead of
    // the first command buffer so a cold run overlaps SSD I/O instead of
    // serializing thousands of GPU page faults.  MADV_WILLNEED is a one-shot
    // hint; MADV_RANDOM above remains appropriate once the model is resident.
    if (prefetch && !std::getenv("NEUTRON_NO_SSD_PREFETCH")) madvise(map_, size_, MADV_WILLNEED);
}

GGUF::~GGUF() { if (map_) munmap(map_,size_); if (fd_>=0) ::close(fd_); }
const Tensor & GGUF::tensor(const std::string & n) const { auto i=tensor_index_.find(n); if(i==tensor_index_.end()) throw std::runtime_error("missing tensor: "+n); return tensors_[i->second]; }
uint64_t GGUF::meta_u64(const std::string & k) const { const auto &v=metadata_.at(k); if(auto p=std::get_if<uint64_t>(&v))return *p; if(auto p=std::get_if<int64_t>(&v))return *p; throw std::runtime_error("metadata is not integer: "+k); }
double GGUF::meta_number(const std::string & k) const { const auto &v=metadata_.at(k); if(auto p=std::get_if<double>(&v))return *p; return double(meta_u64(k)); }

} // namespace neutron::native
