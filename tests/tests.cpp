#include "neutron/config.hpp"
#include "neutron/prompt.hpp"

#include <cassert>
#include <iostream>

using nlohmann::json;

int main() {
    using namespace neutron;
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
    assert(render_gemma4_prompt(multimodal).find("Describe <|image|>") != std::string::npos);

    char arg0[] = "neutron-star";
    char arg1[] = "--ctx";
    char arg2[] = "4096";
    char * argv[] = {arg0, arg1, arg2};
    auto c = Config::from_args(3, argv);
    assert(c.context == 4096);

    std::cout << "all tests passed\n";
}
