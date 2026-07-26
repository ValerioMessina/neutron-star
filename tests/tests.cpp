#include "neutron/config.hpp"
#include "neutron/native/kv_cache.hpp"
#include "neutron/prompt.hpp"
#include "neutron/speculative.hpp"

#include <cassert>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using nlohmann::json;

int main() {
    using namespace neutron;
    assert(Config{}.kv_cache);
    assert(Config{}.sparse_swa);
    assert(Config{}.sparse_swa_stride == 2);
    assert(!Config{}.exact_ffn);
    assert(json_to_gemma_value("hello") == "<|\"|>hello<|\"|>");
    assert(json_to_gemma_value(json{{"city", "Rome"}}) == "{city:<|\"|>Rome<|\"|>}");

    const json messages = json::array({
        {{"role", "system"}, {"content", "Be concise."}},
        {{"role", "user"}, {"content", "Hello"}}
    });
    const auto prompt = render_gemma4_prompt(messages);
    assert(prompt == "<|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n");

    const json tools = json::array({{{"type", "function"}, {"function", {
        {"name", "weather"}, {"description", "Weather lookup"},
        {"parameters", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}}}
    }}}});
    const auto tool_prompt = render_gemma4_prompt(json::array({{{"role", "user"}, {"content", "Weather?"}}}), tools);
    assert(tool_prompt.find("<|tool>declaration:weather") != std::string::npos);
    assert(tool_prompt.find("<|turn>model\n") != std::string::npos);
    const json anthropic_tools = json::array({{{"name", "weather"}, {"input_schema", {{"type", "object"}, {"required", json::array({"city"})}}}}});
    assert(render_gemma4_prompt(json::array({{{"role", "user"}, {"content", "Weather?"}}}), anthropic_tools).find("required:[<|\"|>city<|\"|>]") != std::string::npos);

    const json multimodal = json::array({{{"role", "user"}, {"content", json::array({
        {{"type", "text"}, {"text", "Describe "}},
        {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,x"}}}}
    })}}});
    assert(render_gemma4_prompt(multimodal).find("Describe <__media__>") != std::string::npos);

    char arg0[] = "neutron-star";
    char arg1[] = "--ctx";
    char arg2[] = "4096";
    char arg3[] = "--sparse-context";
    char arg4[] = "--sparse-window";
    char arg5[] = "2048";
    char arg6[] = "--kv-cache-dir";
    char arg7[] = "/tmp/neutron-test-cache";
    char arg8[] = "--kv-cache-entries";
    char arg9[] = "3";
    char arg10[] = "--sparse-swa";
    char arg11[] = "--sparse-swa-recent";
    char arg12[] = "128";
    char arg13[] = "--sparse-swa-stride";
    char arg14[] = "8";
    char * argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                     arg9, arg10, arg11, arg12, arg13, arg14};
    auto c = Config::from_args(15, argv);
    assert(c.context == 4096);
    assert(c.batch == 1024);
    assert(c.sparse_context);
    assert(c.sparse_context_window == 2048);
    assert(c.kv_cache_dir == "/tmp/neutron-test-cache");
    assert(c.kv_cache_entries == 3);
    assert(c.kv_cache);
    assert(c.sparse_swa);
    assert(c.sparse_swa_recent == 128);
    assert(c.sparse_swa_stride == 8);
    char no_cache[] = "--no-kv-cache";
    char * no_cache_argv[] = {arg0, no_cache};
    assert(!Config::from_args(2, no_cache_argv).kv_cache);
    char bad1[] = "--sparse-sink";
    char bad2[] = "-1";
    char * bad_argv[] = {arg0, bad1, bad2};
    bool rejected_sparse_range = false;
    try {
        (void)Config::from_args(3, bad_argv);
    } catch (const std::runtime_error&) {
        rejected_sparse_range = true;
    }
    assert(rejected_sparse_range);

    const auto cache_dir = std::filesystem::temp_directory_path() /
        ("neutron-kv-index-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(cache_dir);
    {
        native::KvCacheIndex index(cache_dir.string(), 1234, 5678, 4096, 1024, 2);
        const std::vector<int32_t> cache_tokens{1,2,3,4};
        const std::vector<native::KvCacheLayer> cache_layers{
            {0,16,2,1,4,0,16,0,16}
        };
        const std::vector<std::byte> cache_payload(32,std::byte{0x2a});
        index.store(cache_tokens,cache_layers,cache_payload.data(),cache_payload.size());
        const std::vector<int32_t> extended{1,2,3,4,5};
        auto hit = index.find_longest(extended,0);
        assert(hit && hit->token_count() == cache_tokens.size());
        assert(std::equal(hit->tokens().begin(),hit->tokens().end(),cache_tokens.begin()));
        assert(hit->layers().size() == 1 && hit->layers()[0].slots == 4);
        const std::vector<int32_t> divergent{1,2,9,4,5};
        assert(!index.find_longest(divergent,0));
    }
    std::filesystem::remove_all(cache_dir);

    const int32_t draft[] = {10, 11, 12};
    const int32_t rejected[] = {10, 11, 20, 21};
    const auto reject = verify_mtp_greedy(draft, rejected);
    assert(reject.accepted == 2 && reject.next_token == 20);
    const int32_t accepted[] = {10, 11, 12, 13};
    const auto accept = verify_mtp_greedy(draft, accepted);
    assert(accept.accepted == 3 && accept.next_token == 13);

    std::cout << "all tests passed\n";
}
