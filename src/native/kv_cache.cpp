#include "neutron/native/kv_cache.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace neutron::native {
namespace {
constexpr std::array<char,8> kMagic{'N','S','K','V','C','0','0','1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kHeaderBytes = 16384;

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t layer_count;
    uint32_t token_count;
    uint32_t context;
    uint32_t batch;
    uint64_t model_size;
    uint64_t model_fingerprint;
    uint64_t token_hash;
    uint64_t payload_bytes;
    uint64_t file_size;
    uint64_t reserved[3];
};

static_assert(sizeof(Header) == 96);

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t candidate_key(uint32_t count, uint64_t hash) {
    hash ^= uint64_t(count) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

bool parse_u32(std::string_view text, uint32_t & value) {
    const auto [end,error] = std::from_chars(text.data(), text.data()+text.size(), value);
    return error == std::errc{} && end == text.data()+text.size();
}

bool parse_hex64(std::string_view text, uint64_t & value) {
    const auto [end,error] = std::from_chars(text.data(), text.data()+text.size(), value, 16);
    return error == std::errc{} && end == text.data()+text.size();
}

void write_all(int fd, const void * data, size_t bytes, uint64_t offset) {
    const auto * cursor = static_cast<const std::byte *>(data);
    while (bytes) {
        const ssize_t written = ::pwrite(fd, cursor, bytes, static_cast<off_t>(offset));
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("KV cache write failed: " + std::string(std::strerror(errno)));
        }
        if (written == 0) throw std::runtime_error("KV cache write made no progress");
        cursor += written;
        bytes -= static_cast<size_t>(written);
        offset += static_cast<uint64_t>(written);
    }
}

bool regular_file(const std::filesystem::directory_entry & entry) {
    std::error_code error;
    return entry.is_regular_file(error) && !error;
}
} // namespace

KvCacheMapping::~KvCacheMapping() {
    if (map_) ::munmap(map_, size_);
    if (fd_ >= 0) ::close(fd_);
}

KvCacheIndex::KvCacheIndex(std::string directory, uint64_t model_size,
                           uint64_t model_fingerprint, uint32_t context,
                           uint32_t batch, uint32_t max_entries)
    : directory_(std::move(directory)), model_size_(model_size),
      model_fingerprint_(model_fingerprint), context_(context), batch_(batch),
      max_entries_(max_entries) {
    if (directory_.empty()) return;
    std::filesystem::create_directories(directory_);
    refresh();
    prune();
}

uint64_t KvCacheIndex::token_hash(std::span<const int32_t> tokens) {
    uint64_t hash = 1469598103934665603ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    for (int32_t token : tokens) {
        const uint32_t value = static_cast<uint32_t>(token);
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            hash ^= uint8_t(value >> shift);
            hash *= prime;
        }
    }
    return hash;
}

std::string KvCacheIndex::file_prefix() const {
    return "nskv1-" + hex64(model_fingerprint_) + "-";
}

void KvCacheIndex::refresh() {
    candidates_.clear();
    if (directory_.empty()) return;
    const std::string prefix = file_prefix();
    std::error_code error;
    for (const auto & entry : std::filesystem::directory_iterator(directory_, error)) {
        if (error || !regular_file(entry)) continue;
        const std::string name = entry.path().filename().string();
        if (!name.starts_with(prefix) || !name.ends_with(".nskv")) continue;
        const std::string_view body(name.data()+prefix.size(),
                                    name.size()-prefix.size()-5);
        const size_t split = body.find('-');
        if (split == std::string_view::npos) continue;
        uint32_t count = 0;
        uint64_t hash = 0;
        if (!parse_u32(body.substr(0,split), count) ||
            !parse_hex64(body.substr(split+1), hash) || count == 0 ||
            count > context_) continue;
        Candidate candidate{count,hash,entry.path()};
        candidates_.emplace(candidate_key(count,hash), std::move(candidate));
    }
}

std::unique_ptr<KvCacheMapping> KvCacheIndex::open_candidate(
    const Candidate & candidate, std::span<const int32_t> input) const {
    auto result = std::unique_ptr<KvCacheMapping>(new KvCacheMapping);
    result->fd_ = ::open(candidate.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (result->fd_ < 0) return {};
    struct stat st{};
    if (::fstat(result->fd_, &st) != 0 ||
        st.st_size < static_cast<off_t>(kHeaderBytes) ||
        st.st_size % kHeaderBytes != 0) return {};
    result->size_ = static_cast<size_t>(st.st_size);
    result->map_ = static_cast<std::byte *>(
        ::mmap(nullptr, result->size_, PROT_READ, MAP_PRIVATE, result->fd_, 0));
    if (result->map_ == MAP_FAILED) {
        result->map_ = nullptr;
        return {};
    }
    const auto & header = *reinterpret_cast<const Header *>(result->map_);
    if (!std::equal(kMagic.begin(), kMagic.end(), header.magic) ||
        header.version != kVersion || header.header_bytes != kHeaderBytes ||
        header.token_count != candidate.token_count ||
        header.token_hash != candidate.token_hash ||
        header.context != context_ || header.batch != batch_ ||
        header.model_size != model_size_ ||
        header.model_fingerprint != model_fingerprint_ ||
        header.file_size != result->size_ ||
        header.payload_bytes > header.file_size-header.header_bytes ||
        sizeof(Header)+uint64_t(header.layer_count)*sizeof(KvCacheLayer)+
            uint64_t(header.token_count)*sizeof(int32_t) > header.header_bytes ||
        header.token_count > input.size()) return {};
    const auto * layer = reinterpret_cast<const KvCacheLayer *>(
        result->map_ + sizeof(Header));
    const auto * token = reinterpret_cast<const int32_t *>(
        result->map_ + sizeof(Header) +
        size_t(header.layer_count)*sizeof(KvCacheLayer));
    result->layers_ = {layer,header.layer_count};
    result->tokens_ = {token,header.token_count};
    result->payload_offset_ = header.header_bytes;
    if (!std::equal(result->tokens_.begin(), result->tokens_.end(), input.begin()))
        return {};
    for (const auto & item : result->layers_) {
        if (item.heads && item.bytes_per_head > header.payload_bytes/item.heads)
            return {};
        const uint64_t bytes = item.bytes_per_head*item.heads;
        if (item.reserved || !item.heads || !item.dim || item.slots > item.capacity ||
            item.key_offset > header.payload_bytes ||
            bytes > header.payload_bytes-item.key_offset ||
            item.value_offset > header.payload_bytes ||
            bytes > header.payload_bytes-item.value_offset) return {};
    }
    ::madvise(result->map_, result->size_, MADV_SEQUENTIAL);
    return result;
}

std::unique_ptr<KvCacheMapping> KvCacheIndex::find_longest(
    std::span<const int32_t> input, size_t longer_than) const {
    const std::lock_guard lock(mutex_);
    if (directory_.empty() || input.empty() || longer_than >= input.size()) return {};
    std::vector<uint64_t> hashes(input.size()+1);
    hashes[0] = token_hash({});
    uint64_t hash = hashes[0];
    constexpr uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t value = static_cast<uint32_t>(input[i]);
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            hash ^= uint8_t(value >> shift);
            hash *= prime;
        }
        hashes[i+1] = hash;
    }
    for (size_t count = input.size(); count > longer_than; --count) {
        const uint64_t key = candidate_key(static_cast<uint32_t>(count),hashes[count]);
        const auto [begin,end] = candidates_.equal_range(key);
        for (auto it = begin; it != end; ++it)
            if (it->second.token_count == count &&
                it->second.token_hash == hashes[count])
                if (auto mapping = open_candidate(it->second,input)) return mapping;
    }
    return {};
}

void KvCacheIndex::store(std::span<const int32_t> tokens,
                         std::span<const KvCacheLayer> layers,
                         const void * payload, uint64_t payload_bytes) {
    if (directory_.empty() || tokens.empty() || !payload || !payload_bytes) return;
    if (tokens.size() > context_ || tokens.size() > UINT32_MAX ||
        layers.size() > UINT32_MAX ||
        sizeof(Header)+layers.size_bytes()+tokens.size_bytes() > kHeaderBytes)
        throw std::runtime_error("KV cache snapshot metadata exceeds its bounds");
    const uint64_t hash = token_hash(tokens);
    const std::string name = file_prefix()+std::to_string(tokens.size())+"-"+
        hex64(hash)+".nskv";
    const std::filesystem::path final =
        std::filesystem::path(directory_)/name;
    const std::filesystem::path temporary =
        final.string()+".tmp-"+std::to_string(::getpid());
    const uint64_t file_size = kHeaderBytes+align_up(payload_bytes,kHeaderBytes);
    Header header{};
    std::copy(kMagic.begin(), kMagic.end(), header.magic);
    header.version = kVersion;
    header.header_bytes = kHeaderBytes;
    header.layer_count = static_cast<uint32_t>(layers.size());
    header.token_count = static_cast<uint32_t>(tokens.size());
    header.context = context_;
    header.batch = batch_;
    header.model_size = model_size_;
    header.model_fingerprint = model_fingerprint_;
    header.token_hash = hash;
    header.payload_bytes = payload_bytes;
    header.file_size = file_size;

    int fd = ::open(temporary.c_str(),
        O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0644);
    if (fd < 0)
        throw std::runtime_error("cannot create KV cache snapshot: "+
                                 std::string(std::strerror(errno)));
    try {
        if (::ftruncate(fd,static_cast<off_t>(file_size)) != 0)
            throw std::runtime_error("cannot size KV cache snapshot");
        std::array<std::byte,kHeaderBytes> page{};
        size_t cursor = 0;
        std::memcpy(page.data()+cursor,&header,sizeof(header));
        cursor += sizeof(header);
        std::memcpy(page.data()+cursor,layers.data(),layers.size_bytes());
        cursor += layers.size_bytes();
        std::memcpy(page.data()+cursor,tokens.data(),tokens.size_bytes());
        write_all(fd,page.data(),page.size(),0);
        write_all(fd,payload,payload_bytes,kHeaderBytes);
        if (::fsync(fd) != 0) throw std::runtime_error("KV cache fsync failed");
        ::close(fd); fd = -1;
        if (::rename(temporary.c_str(),final.c_str()) != 0)
            throw std::runtime_error("KV cache rename failed: "+
                                     std::string(std::strerror(errno)));
    } catch (...) {
        if (fd >= 0) ::close(fd);
        ::unlink(temporary.c_str());
        throw;
    }
    {
        const std::lock_guard lock(mutex_);
        refresh();
        prune();
    }
}

void KvCacheIndex::prune() {
    if (!max_entries_ || candidates_.size() <= max_entries_) return;
    struct Item {
        std::filesystem::file_time_type time;
        std::filesystem::path path;
    };
    std::vector<Item> items;
    items.reserve(candidates_.size());
    std::error_code error;
    for (const auto & [_,candidate] : candidates_) {
        const auto time = std::filesystem::last_write_time(candidate.path,error);
        if (!error) items.push_back({time,candidate.path});
        error.clear();
    }
    std::sort(items.begin(),items.end(),
              [](const Item & a,const Item & b){return a.time>b.time;});
    for (size_t i = max_entries_; i < items.size(); ++i)
        std::filesystem::remove(items[i].path,error);
    refresh();
}

} // namespace neutron::native
