#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace neutron {

struct RenderOptions {
    bool thinking = false;
    bool add_model_turn = true;
};

std::string render_gemma4_prompt(const nlohmann::json & messages,
                                 const nlohmann::json & tools = nlohmann::json::array(),
                                 RenderOptions options = {});
std::string extract_text_content(const nlohmann::json & content);
std::string json_to_gemma_value(const nlohmann::json & value);

} // namespace neutron
