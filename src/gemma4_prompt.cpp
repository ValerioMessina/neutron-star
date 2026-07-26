#include "neutron/prompt.hpp"

#include <sstream>
#include <stdexcept>

namespace neutron {

std::string extract_text_content(const nlohmann::json & content) {
    if (content.is_null()) return {};
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array()) throw std::runtime_error("message content must be a string or array");
    std::string out;
    for (const auto & part : content) {
        const auto type = part.value("type", "text");
        if (type == "text" || type == "input_text") out += part.value("text", "");
        else if (type == "image_url" || type == "input_image" ||
                 type == "input_audio" || type == "input_video" ||
                 type == "video_url" || type == "image" ||
                 type == "audio" || type == "video") out += "<__media__>";
    }
    return out;
}

std::string json_to_gemma_value(const nlohmann::json & v) {
    if (v.is_string()) return "<|\"|>" + v.get<std::string>() + "<|\"|>";
    if (v.is_null()) return "null";
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number()) return v.dump();
    if (v.is_array()) {
        std::string out = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += ',';
            out += json_to_gemma_value(v[i]);
        }
        return out + ']';
    }
    std::string out = "{";
    bool first = true;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (!first) out += ',';
        first = false;
        out += it.key() + ':' + json_to_gemma_value(it.value());
    }
    return out + '}';
}

static std::string render_tools(const nlohmann::json & tools) {
    std::string out;
    if (!tools.is_array()) throw std::runtime_error("tools must be an array");
    for (const auto & entry : tools) {
        const auto & f = entry.contains("function") ? entry.at("function") : entry;
        const std::string name = f.value("name", "");
        if (name.empty()) throw std::runtime_error("tool name is required");
        nlohmann::json declaration = {
            {"description", f.value("description", "")},
            {"parameters", f.value("parameters", f.value("input_schema", nlohmann::json::object()))}
        };
        out += "<|tool>declaration:" + name + json_to_gemma_value(declaration) + "<tool|>";
    }
    return out;
}

std::string render_gemma4_prompt(const nlohmann::json & messages,
                                 const nlohmann::json & tools,
                                 RenderOptions options) {
    if (!messages.is_array() || messages.empty()) throw std::runtime_error("messages must be a non-empty array");
    std::string system;
    nlohmann::json turns = nlohmann::json::array();
    for (const auto & m : messages) {
        const std::string role = m.value("role", "");
        if (role == "system" || role == "developer") {
            if (!system.empty()) system += '\n';
            system += extract_text_content(m.value("content", nlohmann::json()));
        } else {
            turns.push_back(m);
        }
    }

    std::string out;
    const std::string tool_text = render_tools(tools);
    if (!system.empty() || options.thinking || !tool_text.empty()) {
        out += "<|turn>system\n";
        if (options.thinking) out += "<|think|>";
        out += system + tool_text + "<turn|>\n";
    }
    for (const auto & m : turns) {
        std::string role = m.value("role", "");
        if (role == "tool") role = "model";
        if (role == "assistant") role = "model";
        if (role != "user" && role != "model") throw std::runtime_error("unsupported message role");
        out += "<|turn>" + role + "\n";
        if (m.contains("tool_calls")) {
            for (const auto & tc : m.at("tool_calls")) {
                const auto & f = tc.at("function");
                nlohmann::json args = f.value("arguments", nlohmann::json::object());
                if (args.is_string()) args = nlohmann::json::parse(args.get<std::string>());
                out += "<|tool_call>call:" + f.at("name").get<std::string>() + json_to_gemma_value(args) + "<tool_call|>";
            }
        }
        if (m.value("role", "") == "tool") {
            out += "<|tool_response>response:" + m.value("name", "tool") + json_to_gemma_value(m.value("content", nlohmann::json())) + "<tool_response|>";
        } else {
            out += extract_text_content(m.value("content", nlohmann::json()));
        }
        out += "<turn|>\n";
    }
    if (options.add_model_turn) out += "<|turn>model\n";
    return out;
}

} // namespace neutron
