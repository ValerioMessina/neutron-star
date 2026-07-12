#include "neutron/engine.hpp"

#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace neutron {
namespace {
using Clock = std::chrono::steady_clock;

void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_CONT) std::cerr << text;
}

class LlamaEngine final : public Engine {
public:
    explicit LlamaEngine(const Config & config) : config_(config) {
        if (!config.verbose) llama_log_set(quiet_log, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = config.gpu_layers;
        mp.use_mmap = true;
        mp.use_mlock = config.mlock;
        model_ = llama_model_load_from_file(config.model_path.c_str(), mp);
        if (!model_) throw std::runtime_error("cannot load GGUF: " + config.model_path);
        vocab_ = llama_model_get_vocab(model_);

        auto cp = llama_context_default_params();
        cp.n_ctx = config.context;
        cp.n_batch = config.batch;
        cp.n_ubatch = config.ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = config.threads;
        cp.n_threads_batch = config.threads;
        cp.flash_attn_type = config.flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.no_perf = false;
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) {
            llama_model_free(model_);
            model_ = nullptr;
            throw std::runtime_error("cannot allocate context; reduce --ctx or --batch");
        }
        char desc[256]{};
        llama_model_desc(model_, desc, sizeof(desc));
        description_ = desc;
        restore_active_state();
    }

    ~LlamaEngine() override {
        if (ctx_) llama_free(ctx_);
        if (model_) llama_model_free(model_);
        llama_backend_free();
    }

    GenerationResult generate(const std::string & prompt,
                              const SamplingParams & params,
                              const TokenCallback & callback) override {
        std::lock_guard lock(mu_); // one mutable graph/KV session, deliberately serialized
        if (params.max_tokens < 0) throw std::runtime_error("max_tokens cannot be negative");
        auto tokens = tokenize(prompt);
        if (tokens.size() + static_cast<size_t>(params.max_tokens) > config_.context) {
            throw std::runtime_error("prompt plus max_tokens exceeds configured context");
        }

        GenerationResult result;
        result.stats.prompt_tokens = tokens.size();
        size_t prefix = common_prefix(tokens, cached_tokens_);
        // The sequence checkpoint stores KV tensors, not a reusable final logits row.
        // Replay one token on an exact hit so sampling always observes fresh logits.
        if (prefix == tokens.size() && prefix > 0) --prefix;
        auto memory = llama_get_memory(ctx_);
        if (prefix < cached_tokens_.size()) {
            if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(prefix), -1)) {
                llama_memory_clear(memory, true);
                prefix = 0;
            }
            cached_tokens_.resize(prefix);
        }
        result.stats.cached_tokens = prefix;

        const auto prefill_start = Clock::now();
        for (size_t off = prefix; off < tokens.size();) {
            const size_t count = std::min<size_t>(config_.batch, tokens.size() - off);
            auto batch = llama_batch_get_one(tokens.data() + off, static_cast<int32_t>(count));
            if (llama_decode(ctx_, batch) != 0) throw std::runtime_error("prefill decode failed");
            cached_tokens_.insert(cached_tokens_.end(), tokens.begin() + off, tokens.begin() + off + count);
            off += count;
        }
        result.stats.prefill_ms = elapsed_ms(prefill_start);

        auto sampler = make_sampler(params);
        for (llama_token t : tokens) llama_sampler_accept(sampler, t);
        const auto generation_start = Clock::now();
        std::string pending;
        bool cancelled = false;
        bool stopped = false;
        for (int i = 0; i < params.max_tokens; ++i) {
            llama_token token = llama_sampler_sample(sampler, ctx_, -1);
            if (llama_vocab_is_eog(vocab_, token)) {
                stopped = true;
                break;
            }
            const std::string piece = token_piece(token);
            pending += piece;
            result.stats.generated_tokens++;

            size_t stop_at = std::string::npos;
            for (const auto & stop : params.stop) {
                if (!stop.empty()) stop_at = std::min(stop_at, pending.find(stop));
            }
            if (stop_at != std::string::npos) {
                const std::string emit = pending.substr(0, stop_at);
                result.text += emit;
                if (callback && !emit.empty() && !callback(emit)) cancelled = true;
                pending.clear();
                stopped = true;
                break;
            }

            size_t retain = 0;
            for (const auto & stop : params.stop) {
                const size_t max = std::min(stop.size(), pending.size());
                for (size_t n = 1; n <= max; ++n) {
                    if (pending.compare(pending.size() - n, n, stop, 0, n) == 0) retain = std::max(retain, n);
                }
            }
            const std::string emit = pending.substr(0, pending.size() - retain);
            pending.erase(0, pending.size() - retain);
            result.text += emit;
            if (callback && !emit.empty() && !callback(emit)) {
                cancelled = true;
                break;
            }

            auto batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx_, batch) != 0) throw std::runtime_error("token decode failed");
            cached_tokens_.push_back(token);
        }
        if (!cancelled && !stopped && !pending.empty()) {
            result.text += pending;
            if (callback && !callback(pending)) cancelled = true;
        }
        result.stats.generation_ms = elapsed_ms(generation_start);
        result.stats.finish_reason = cancelled ? "cancelled" : (stopped ? "stop" : "length");
        llama_sampler_free(sampler);
        persist_active_state();
        return result;
    }

    std::string model_description() const override { return description_; }
    uint64_t model_size() const override { return llama_model_size(model_); }
    uint32_t context_size() const override { return llama_n_ctx(ctx_); }

private:
    static double elapsed_ms(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    static size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return n;
    }

    std::vector<llama_token> tokenize(const std::string & text) const {
        int32_t n = llama_tokenize(vocab_, text.data(), text.size(), nullptr, 0, true, true);
        if (n >= 0) return {};
        std::vector<llama_token> out(static_cast<size_t>(-n));
        n = llama_tokenize(vocab_, text.data(), text.size(), out.data(), out.size(), true, true);
        if (n < 0) throw std::runtime_error("tokenization failed");
        out.resize(n);
        return out;
    }

    std::string token_piece(llama_token token) const {
        std::string out(64, '\0');
        int32_t n = llama_token_to_piece(vocab_, token, out.data(), out.size(), 0, false);
        if (n < 0) {
            out.resize(static_cast<size_t>(-n));
            n = llama_token_to_piece(vocab_, token, out.data(), out.size(), 0, false);
        }
        if (n < 0) throw std::runtime_error("detokenization failed");
        out.resize(n);
        return out;
    }

    llama_sampler * make_sampler(const SamplingParams & p) const {
        auto sp = llama_sampler_chain_default_params();
        sp.no_perf = true;
        llama_sampler * chain = llama_sampler_chain_init(sp);
        if (p.repeat_penalty != 1.0F) {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, p.repeat_penalty, 0, 0));
        }
        if (p.temperature <= 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, 1));
            if (p.min_p > 0) llama_sampler_chain_add(chain, llama_sampler_init_min_p(p.min_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(p.temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(p.seed));
        }
        return chain;
    }

    std::filesystem::path state_path() const {
        return std::filesystem::path(config_.kv_cache_dir) / "active.kv";
    }

    void restore_active_state() {
        if (config_.kv_cache_dir.empty()) return;
        std::filesystem::create_directories(config_.kv_cache_dir);
        if (!std::filesystem::exists(state_path())) return;
        cached_tokens_.resize(config_.context);
        size_t count = 0;
        if (llama_state_seq_load_file(ctx_, state_path().c_str(), 0, cached_tokens_.data(), cached_tokens_.size(), &count) == 0) {
            cached_tokens_.clear();
            return;
        }
        cached_tokens_.resize(count);
        std::cerr << "restored KV checkpoint: " << count << " tokens\n";
    }

    void persist_active_state() {
        if (config_.kv_cache_dir.empty() || cached_tokens_.empty()) return;
        const auto tmp = state_path().string() + ".tmp";
        if (llama_state_seq_save_file(ctx_, tmp.c_str(), 0, cached_tokens_.data(), cached_tokens_.size()) == 0) return;
        std::error_code ec;
        std::filesystem::rename(tmp, state_path(), ec);
        if (ec) {
            std::filesystem::remove(state_path(), ec);
            std::filesystem::rename(tmp, state_path(), ec);
        }
    }

    Config config_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    std::string description_;
    std::vector<llama_token> cached_tokens_;
    std::mutex mu_;
};
} // namespace

std::unique_ptr<Engine> make_llama_engine(const Config & config) {
    return std::make_unique<LlamaEngine>(config);
}

} // namespace neutron
