#include "neutron/native/multimodal.hpp"

#include <ggml-backend.h>
#include <llama.h>
#include <mtmd-helper.h>
#include <mtmd.h>

#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace neutron::native {
namespace {

void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_CONT) {
        std::cerr << text;
    }
}

std::vector<unsigned char> decode_base64(std::string_view input) {
    static constexpr signed char table[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
        -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    std::vector<unsigned char> out;
    out.reserve(input.size() * 3 / 4);
    uint32_t value = 0;
    int bits = -8;
    for (unsigned char c : input) {
        if (std::isspace(c)) continue;
        if (c == '=') break;
        const int digit = table[c];
        if (digit < 0) throw std::runtime_error("invalid base64 media data");
        value = (value << 6) | static_cast<uint32_t>(digit);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<unsigned char>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

struct LoadedBitmap {
    mtmd_bitmap * bitmap = nullptr;
    mtmd_helper_video * video = nullptr;

    LoadedBitmap() = default;
    LoadedBitmap(const LoadedBitmap &) = delete;
    LoadedBitmap & operator=(const LoadedBitmap &) = delete;
    LoadedBitmap(LoadedBitmap && other) noexcept
        : bitmap(std::exchange(other.bitmap, nullptr)),
          video(std::exchange(other.video, nullptr)) {}
    ~LoadedBitmap() {
        if (bitmap) mtmd_bitmap_free(bitmap);
        if (video) mtmd_helper_video_free(video);
    }
};

} // namespace

struct MultimodalProcessor::Impl {
    llama_model * model = nullptr;
    mtmd_context * context = nullptr;
    std::string media_marker;
    bool vision = false;
    bool audio = false;
    bool video = false;

    Impl(const std::string & model_path, const std::string & mmproj_path) {
        llama_log_set(quiet_log, nullptr);
        mtmd_helper_log_set(quiet_log, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        try {
            auto model_params = llama_model_default_params();
            model_params.vocab_only = true;
            model_params.use_mmap = true;
            model = llama_model_load_from_file(model_path.c_str(), model_params);
            if (!model) {
                throw std::runtime_error("cannot load Gemma vocabulary for multimodal input");
            }
            auto params = mtmd_context_params_default();
            params.use_gpu = true;
            params.print_timings = false;
            params.warmup = false;
            params.image_max_tokens = 256;
            context = mtmd_init_from_file(mmproj_path.c_str(), model, params);
            if (!context) {
                throw std::runtime_error("cannot load multimodal projector: " + mmproj_path);
            }
            vision = mtmd_support_vision(context);
            audio = mtmd_support_audio(context);
            video = vision && mtmd_helper_support_video(context);
            media_marker = mtmd_get_marker(context);
            if (media_marker.empty()) media_marker = mtmd_default_marker();
        } catch (...) {
            if (context) mtmd_free(context);
            if (model) llama_model_free(model);
            llama_backend_free();
            throw;
        }
    }

    ~Impl() {
        if (context) mtmd_free(context);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    LoadedBitmap load(const MediaInput & media) {
        if (!supports(media.type)) {
            throw std::runtime_error("multimodal projector does not support " + media.type);
        }
        if (media.source.rfind("http://", 0) == 0 || media.source.rfind("https://", 0) == 0) {
            throw std::runtime_error("remote media URLs are not supported; use a data URL or local path");
        }

        mtmd_helper_bitmap_wrapper wrapper{};
        if (media.source.rfind("data:", 0) == 0) {
            const size_t comma = media.source.find(',');
            if (comma == std::string::npos ||
                media.source.substr(0, comma).find(";base64") == std::string::npos) {
                throw std::runtime_error("media data URL must use base64 encoding");
            }
            auto bytes = decode_base64(std::string_view(media.source).substr(comma + 1));
            wrapper = mtmd_helper_bitmap_init_from_buf(context, bytes.data(), bytes.size(), false);
        } else {
            std::string path = media.source;
            if (path.rfind("file://", 0) == 0) path.erase(0, 7);
            if (!std::filesystem::is_regular_file(path)) {
                throw std::runtime_error("media file not found: " + path);
            }
            wrapper = mtmd_helper_bitmap_init_from_file(context, path.c_str(), false);
        }
        if (!wrapper.bitmap) throw std::runtime_error("cannot decode " + media.type + " input");
        LoadedBitmap out;
        out.bitmap = wrapper.bitmap;
        out.video = wrapper.video_ctx;
        return out;
    }

    bool supports(const std::string & type) const {
        if (type == "image") return vision;
        if (type == "audio") return audio;
        if (type == "video") return video;
        return false;
    }
};

MultimodalProcessor::MultimodalProcessor(const std::string & model_path,
                                         const std::string & mmproj_path)
    : impl_(std::make_unique<Impl>(model_path, mmproj_path)) {}

MultimodalProcessor::~MultimodalProcessor() = default;

std::vector<PromptChunk> MultimodalProcessor::encode(
        const std::string & prompt,
        const std::vector<MediaInput> & media) {
    std::vector<LoadedBitmap> loaded;
    loaded.reserve(media.size());
    std::vector<const mtmd_bitmap *> bitmaps;
    bitmaps.reserve(media.size());
    for (const auto & item : media) {
        loaded.push_back(impl_->load(item));
        bitmaps.push_back(loaded.back().bitmap);
    }

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    mtmd_input_text text{prompt.c_str(), true, true};
    const int32_t result = mtmd_tokenize(
        impl_->context, chunks.ptr.get(), &text, bitmaps.data(), bitmaps.size());
    if (result != 0) {
        throw std::runtime_error("multimodal prompt tokenization failed: " + std::to_string(result));
    }

    std::vector<PromptChunk> output;
    const size_t count = mtmd_input_chunks_size(chunks.ptr.get());
    output.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        const auto type = mtmd_input_chunk_get_type(chunk);
        PromptChunk item;
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
            item.tokens.assign(tokens, tokens + n);
        } else {
            const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
            if (mtmd_input_chunk_get_n_pos(chunk) != static_cast<llama_pos>(n)) {
                throw std::runtime_error("multimodal M-RoPE positions are not supported by the native Gemma path");
            }
            if (mtmd_encode_chunk(impl_->context, chunk) != 0) {
                throw std::runtime_error("multimodal encoder failed");
            }
            const float * embeddings = mtmd_get_output_embd(impl_->context);
            item.embeddings.assign(embeddings, embeddings + n * 3840);
            item.non_causal = mtmd_decode_use_non_causal(impl_->context, chunk);
        }
        if (item.size()) output.push_back(std::move(item));
    }
    return output;
}

bool MultimodalProcessor::supports(const std::string & type) const {
    return impl_->supports(type);
}

const std::string & MultimodalProcessor::marker() const {
    return impl_->media_marker;
}

} // namespace neutron::native
