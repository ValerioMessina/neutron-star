#include "neutron/server.hpp"
#include "neutron/prompt.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace neutron {
namespace {
using json = nlohmann::json;

std::string id(const char * prefix) {
    static std::atomic<uint64_t> sequence{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream out;
    out << prefix << std::hex << now << sequence.fetch_add(1);
    return out.str();
}

int64_t unix_time() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool secure_equal(std::string_view a, std::string_view b) {
    size_t diff = a.size() ^ b.size();
    const size_t n = std::max(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? a[i] : 0;
        const unsigned char y = i < b.size() ? b[i] : 0;
        diff |= x ^ y;
    }
    return diff == 0;
}

json api_error(const std::string & message, const std::string & type = "invalid_request_error") {
    return {{"error", {{"message", message}, {"type", type}, {"code", nullptr}}}};
}

SamplingParams sampling(const json & body, int default_max = 1024) {
    SamplingParams p;
    p.max_tokens = body.value("max_completion_tokens", body.value("max_output_tokens", body.value("max_tokens", default_max)));
    p.temperature = body.value("temperature", 1.0F);
    p.top_p = body.value("top_p", 0.95F);
    p.top_k = body.value("top_k", 64);
    p.min_p = body.value("min_p", 0.0F);
    p.seed = body.value("seed", 0xFFFFFFFFU);
    p.repeat_penalty = body.value("repetition_penalty", 1.0F);
    if (body.contains("stop")) {
        if (body["stop"].is_string()) p.stop.push_back(body["stop"]);
        else if (body["stop"].is_array()) p.stop = body["stop"].get<std::vector<std::string>>();
    }
    if (body.contains("stop_sequences")) p.stop = body["stop_sequences"].get<std::vector<std::string>>();
    if (p.max_tokens < 0 || p.max_tokens > 32768) throw std::runtime_error("max_tokens must be between 0 and 32768");
    if (p.temperature < 0 || p.temperature > 2) throw std::runtime_error("temperature must be between 0 and 2");
    if (p.top_p <= 0 || p.top_p > 1) throw std::runtime_error("top_p must be in (0, 1]");
    return p;
}

json normalized_messages(const json & body) {
    if (!body.contains("messages")) throw std::runtime_error("messages is required");
    json messages = body.at("messages");
    if (body.contains("system")) {
        const auto system = body.at("system");
        messages.insert(messages.begin(), {{"role", "system"}, {"content", system}});
    }
    return messages;
}

json anthropic_messages(const json & body) {
    if (!body.contains("messages") || !body["messages"].is_array()) throw std::runtime_error("messages is required");
    json out = json::array();
    if (body.contains("system")) out.push_back({{"role", "system"}, {"content", body["system"]}});
    std::unordered_map<std::string, std::string> tool_names;
    for (const auto & source : body["messages"]) {
        const std::string role = source.value("role", "");
        const json content = source.value("content", json());
        if (!content.is_array()) { out.push_back(source); continue; }
        std::string text;
        json calls = json::array();
        for (const auto & part : content) {
            const std::string type = part.value("type", "text");
            if (type == "text") text += part.value("text", "");
            else if (type == "image") text += "<|image|>";
            else if (type == "tool_use") {
                const std::string call_id = part.value("id", "");
                const std::string name = part.value("name", "");
                tool_names[call_id] = name;
                calls.push_back({{"id", call_id}, {"type", "function"},
                    {"function", {{"name", name}, {"arguments", part.value("input", json::object()).dump()}}}});
            } else if (type == "tool_result") {
                const std::string call_id = part.value("tool_use_id", "");
                out.push_back({{"role", "tool"}, {"tool_call_id", call_id},
                    {"name", tool_names.contains(call_id) ? tool_names[call_id] : "tool"},
                    {"content", extract_text_content(part.value("content", json()))}});
            }
        }
        if (!text.empty() || !calls.empty()) {
            json message = {{"role", role}, {"content", text}};
            if (!calls.empty()) message["tool_calls"] = calls;
            out.push_back(std::move(message));
        }
    }
    return out;
}

std::string gemma_args_to_json(std::string args) {
    const std::string quote = "<|\"|>";
    size_t pos = 0;
    bool in_string = false;
    while ((pos = args.find(quote, pos)) != std::string::npos) {
        args.replace(pos, quote.size(), "\"");
        in_string = !in_string;
        ++pos;
    }
    std::string out;
    in_string = false;
    for (size_t i = 0; i < args.size();) {
        if (args[i] == '"') { in_string = !in_string; out += args[i++]; continue; }
        const bool key_start = !in_string && (i == 0 || args[i - 1] == '{' || args[i - 1] == ',') &&
                               (std::isalpha(static_cast<unsigned char>(args[i])) || args[i] == '_');
        if (key_start) {
            size_t j = i + 1;
            while (j < args.size() && (std::isalnum(static_cast<unsigned char>(args[j])) || args[j] == '_' || args[j] == '-')) ++j;
            if (j < args.size() && args[j] == ':') {
                out += '"'; out.append(args, i, j - i); out += '"'; i = j; continue;
            }
        }
        out += args[i++];
    }
    return out;
}

struct ParsedOutput { std::string text; json calls = json::array(); };

ParsedOutput parse_output(const std::string & raw) {
    ParsedOutput p;
    size_t cursor = 0;
    const std::string open = "<|tool_call>call:";
    const std::string close = "<tool_call|>";
    while (true) {
        const size_t start = raw.find(open, cursor);
        if (start == std::string::npos) { p.text += raw.substr(cursor); break; }
        p.text += raw.substr(cursor, start - cursor);
        const size_t end = raw.find(close, start + open.size());
        if (end == std::string::npos) { p.text += raw.substr(start); break; }
        const std::string call = raw.substr(start + open.size(), end - start - open.size());
        const size_t brace = call.find('{');
        const std::string name = call.substr(0, brace);
        const std::string args = brace == std::string::npos ? "{}" : gemma_args_to_json(call.substr(brace));
        p.calls.push_back({{"id", id("call_")}, {"type", "function"}, {"function", {{"name", name}, {"arguments", args}}}});
        cursor = end + close.size();
    }
    const size_t channel = p.text.find("<channel|>");
    if (channel != std::string::npos) p.text.erase(0, channel + std::string("<channel|>").size());
    return p;
}

std::string sse(const json & event) { return "data: " + event.dump() + "\n\n"; }

bool write(httplib::DataSink & sink, const std::string & data) {
    return sink.is_writable() && sink.write(data.data(), data.size());
}

class VisibleStream {
public:
    explicit VisibleStream(std::function<bool(const std::string &)> emit) : emit_(std::move(emit)) {}
    bool feed(const std::string & bytes) {
        if (decided_) { visible_ += bytes; return emit_(bytes); }
        pending_ += bytes;
        static const std::string open = "<|channel>";
        static const std::string close = "<channel|>";
        if (open.rfind(pending_, 0) == 0) return true;
        if (pending_.rfind(open, 0) == 0) {
            const size_t end = pending_.find(close);
            if (end == std::string::npos) return true;
            pending_.erase(0, end + close.size());
        }
        decided_ = true;
        visible_ += pending_;
        const std::string emit = std::exchange(pending_, {});
        return emit.empty() || emit_(emit);
    }
    bool finish() {
        if (!decided_ && !pending_.empty() && pending_.rfind("<|channel>", 0) != 0) {
            decided_ = true; visible_ += pending_; return emit_(std::exchange(pending_, {}));
        }
        return true;
    }
    const std::string & text() const { return visible_; }
private:
    std::function<bool(const std::string &)> emit_;
    std::string pending_;
    std::string visible_;
    bool decided_ = false;
};

void set_headers(httplib::Response & res, const std::string & request_id) {
    res.set_header("X-Request-Id", request_id);
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Content-Type-Options", "nosniff");
}
} // namespace

int run_server(const Config & config, Engine & engine) {
    httplib::Server server;
    server.set_payload_max_length(8 * 1024 * 1024);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(600, 0);

    server.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (config.api_key.empty() || req.path == "/healthz") return httplib::Server::HandlerResponse::Unhandled;
        std::string supplied;
        if (req.has_header("Authorization")) {
            supplied = req.get_header_value("Authorization");
            if (supplied.rfind("Bearer ", 0) == 0) supplied.erase(0, 7);
        } else if (req.has_header("x-api-key")) supplied = req.get_header_value("x-api-key");
        if (!secure_equal(supplied, config.api_key)) {
            res.status = 401;
            res.set_content(api_error("invalid API key", "authentication_error").dump(), "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server.Get("/healthz", [](const auto &, auto & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    server.Get("/v1/models", [&](const auto &, auto & res) {
        res.set_content(json{{"object", "list"}, {"data", json::array({{
            {"id", config.model_name}, {"object", "model"}, {"created", unix_time()}, {"owned_by", "local"},
            {"context_length", engine.context_size()}, {"bytes", engine.model_size()}
        }})}}.dump(), "application/json");
    });
    server.Get(R"(/v1/models/(.+))", [&](const httplib::Request &, auto & res) {
        res.set_content(json{{"id", config.model_name}, {"object", "model"}, {"owned_by", "local"}}.dump(), "application/json");
    });

    server.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const json body = json::parse(req.body);
            const auto params = sampling(body);
            const auto messages = normalized_messages(body);
            const json tools = body.value("tools", json::array());
            const bool thinking = body.value("thinking", false);
            const std::string prompt = render_gemma4_prompt(messages, tools, {.thinking = thinking});
            const std::string request_id = id("chatcmpl_");
            set_headers(res, request_id);
            if (!body.value("stream", false)) {
                auto generated = engine.generate(prompt, params);
                auto parsed = parse_output(generated.text);
                json message = {{"role", "assistant"}, {"content", parsed.text}};
                if (!parsed.calls.empty()) message["tool_calls"] = parsed.calls;
                const std::string finish = parsed.calls.empty() ? generated.stats.finish_reason : "tool_calls";
                json usage = {
                    {"prompt_tokens", generated.stats.prompt_tokens},
                    {"completion_tokens", generated.stats.generated_tokens},
                    {"total_tokens", generated.stats.prompt_tokens + generated.stats.generated_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", generated.stats.cached_tokens}}}
                };
                json response = {
                    {"id", request_id}, {"object", "chat.completion"}, {"created", unix_time()},
                    {"model", config.model_name},
                    {"choices", json::array({{{"index", 0}, {"message", message}, {"finish_reason", finish}}})},
                    {"usage", usage}
                };
                res.set_content(response.dump(), "application/json");
                return;
            }
            res.set_chunked_content_provider("text/event-stream", [&, prompt, params, tools, request_id](size_t, httplib::DataSink & sink) {
                try {
                    const json first = {{"id", request_id}, {"object", "chat.completion.chunk"}, {"created", unix_time()},
                        {"model", config.model_name}, {"choices", json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}})}};
                    if (!write(sink, sse(first))) return false;
                    const bool buffer_for_tools = !tools.empty();
                    VisibleStream visible([&](const std::string & delta) {
                        if (buffer_for_tools) return sink.is_writable();
                        return write(sink, sse({{"id", request_id}, {"object", "chat.completion.chunk"}, {"created", unix_time()},
                            {"model", config.model_name}, {"choices", json::array({{{"index", 0}, {"delta", {{"content", delta}}}, {"finish_reason", nullptr}}})}}));
                    });
                    auto result = engine.generate(prompt, params, [&](const std::string & delta) { return visible.feed(delta); });
                    visible.finish();
                    auto parsed = parse_output(result.text);
                    if (buffer_for_tools && !parsed.text.empty()) {
                        if (!write(sink, sse({{"id", request_id}, {"object", "chat.completion.chunk"}, {"choices", json::array({{{"index", 0}, {"delta", {{"content", parsed.text}}}, {"finish_reason", nullptr}}})}}))) return false;
                    }
                    json delta = json::object();
                    if (!parsed.calls.empty()) delta["tool_calls"] = parsed.calls;
                    const std::string finish = parsed.calls.empty() ? result.stats.finish_reason : "tool_calls";
                    write(sink, sse({{"id", request_id}, {"object", "chat.completion.chunk"}, {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})},
                        {"usage", {{"prompt_tokens", result.stats.prompt_tokens}, {"completion_tokens", result.stats.generated_tokens}, {"total_tokens", result.stats.prompt_tokens + result.stats.generated_tokens}}}}));
                    write(sink, "data: [DONE]\n\n");
                } catch (const std::exception & e) { write(sink, sse(api_error(e.what(), "server_error"))); }
                sink.done();
                return true;
            });
        } catch (const std::exception & e) {
            res.status = 400; res.set_content(api_error(e.what()).dump(), "application/json");
        }
    });

    server.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const json body = json::parse(req.body);
            if (!body.contains("prompt") || !body["prompt"].is_string()) throw std::runtime_error("prompt string is required");
            auto result = engine.generate(body["prompt"], sampling(body));
            res.set_content(json{{"id", id("cmpl_")}, {"object", "text_completion"}, {"created", unix_time()}, {"model", config.model_name},
                {"choices", json::array({{{"text", result.text}, {"index", 0}, {"finish_reason", result.stats.finish_reason}}})},
                {"usage", {{"prompt_tokens", result.stats.prompt_tokens}, {"completion_tokens", result.stats.generated_tokens}, {"total_tokens", result.stats.prompt_tokens + result.stats.generated_tokens}}}}.dump(), "application/json");
        } catch (const std::exception & e) { res.status = 400; res.set_content(api_error(e.what()).dump(), "application/json"); }
    });

    server.Post("/v1/responses", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const json body = json::parse(req.body);
            json messages = json::array();
            if (body.contains("instructions")) messages.push_back({{"role", "system"}, {"content", body["instructions"]}});
            const json input = body.value("input", json());
            if (input.is_string()) messages.push_back({{"role", "user"}, {"content", input}});
            else if (input.is_array()) for (const auto & item : input) messages.push_back(item);
            else throw std::runtime_error("input must be a string or array");
            const json response_tools = body.value("tools", json::array());
            const std::string prompt = render_gemma4_prompt(messages, response_tools);
            const auto params = sampling(body);
            const std::string rid = id("resp_");
            const std::string mid = id("msg_");
            if (body.value("stream", false)) {
                set_headers(res, rid);
                res.set_chunked_content_provider("text/event-stream", [&, prompt, params, response_tools, rid, mid](size_t, httplib::DataSink & sink) {
                    try {
                        json base = {{"id", rid}, {"object", "response"}, {"created_at", unix_time()}, {"status", "in_progress"}, {"model", config.model_name}, {"output", json::array()}};
                        write(sink, "event: response.created\n" + sse({{"type", "response.created"}, {"response", base}, {"sequence_number", 0}}));
                        write(sink, "event: response.output_item.added\n" + sse({{"type", "response.output_item.added"}, {"output_index", 0}, {"item", {{"id", mid}, {"type", "message"}, {"role", "assistant"}, {"status", "in_progress"}, {"content", json::array()}}}, {"sequence_number", 1}}));
                        write(sink, "event: response.content_part.added\n" + sse({{"type", "response.content_part.added"}, {"item_id", mid}, {"output_index", 0}, {"content_index", 0}, {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", json::array()}}}, {"sequence_number", 2}}));
                        VisibleStream visible([&](const std::string & delta) {
                            return write(sink, "event: response.output_text.delta\n" + sse({{"type", "response.output_text.delta"}, {"item_id", mid}, {"output_index", 0}, {"content_index", 0}, {"delta", delta}}));
                        });
                        auto result = response_tools.empty()
                            ? engine.generate(prompt, params, [&](const std::string & delta) { return visible.feed(delta); })
                            : engine.generate(prompt, params);
                        visible.finish();
                        auto parsed = parse_output(result.text);
                        if (!response_tools.empty() && !parsed.text.empty()) visible.feed(parsed.text);
                        write(sink, "event: response.output_text.done\n" + sse({{"type", "response.output_text.done"}, {"item_id", mid}, {"output_index", 0}, {"content_index", 0}, {"text", visible.text()}}));
                        json done = base; done["status"] = "completed";
                        done["output"] = json::array({{{"id", mid}, {"type", "message"}, {"role", "assistant"}, {"status", "completed"}, {"content", json::array({{{"type", "output_text"}, {"text", visible.text()}, {"annotations", json::array()}}})}}});
                        done["usage"] = {{"input_tokens", result.stats.prompt_tokens}, {"output_tokens", result.stats.generated_tokens}, {"total_tokens", result.stats.prompt_tokens + result.stats.generated_tokens}};
                        for (const auto & call : parsed.calls) done["output"].push_back({{"type", "function_call"}, {"id", call["id"]},
                            {"call_id", call["id"]}, {"name", call["function"]["name"]}, {"arguments", call["function"]["arguments"]}, {"status", "completed"}});
                        write(sink, "event: response.completed\n" + sse({{"type", "response.completed"}, {"response", done}}));
                    } catch (const std::exception & e) { write(sink, "event: error\n" + sse({{"type", "error"}, {"error", {{"type", "server_error"}, {"message", e.what()}}}})); }
                    sink.done(); return true;
                });
                return;
            }
            const auto result = engine.generate(prompt, params);
            auto parsed = parse_output(result.text);
            json output = json::array({{{"id", mid}, {"type", "message"}, {"role", "assistant"},
                {"status", "completed"}, {"content", json::array({{{"type", "output_text"}, {"text", parsed.text}, {"annotations", json::array()}}})}}});
            for (const auto & call : parsed.calls) output.push_back({{"type", "function_call"}, {"id", call["id"]},
                {"call_id", call["id"]}, {"name", call["function"]["name"]}, {"arguments", call["function"]["arguments"]}, {"status", "completed"}});
            res.set_content(json{{"id", rid}, {"object", "response"}, {"created_at", unix_time()}, {"status", "completed"},
                {"model", config.model_name}, {"output", output}, {"usage", {{"input_tokens", result.stats.prompt_tokens}, {"output_tokens", result.stats.generated_tokens}, {"total_tokens", result.stats.prompt_tokens + result.stats.generated_tokens}}}}.dump(), "application/json");
        } catch (const std::exception & e) { res.status = 400; res.set_content(api_error(e.what()).dump(), "application/json"); }
    });

    server.Post("/v1/messages", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const json body = json::parse(req.body);
            const auto params = sampling(body);
            const std::string prompt = render_gemma4_prompt(anthropic_messages(body), body.value("tools", json::array()),
                {.thinking = body.contains("thinking") && body["thinking"].value("type", "disabled") == "enabled"});
            const std::string mid = id("msg_");
            const bool has_tools = !body.value("tools", json::array()).empty();
            if (body.value("stream", false)) {
                set_headers(res, mid);
                res.set_chunked_content_provider("text/event-stream", [&, prompt, params, mid, has_tools](size_t, httplib::DataSink & sink) {
                    try {
                        json message = {{"id", mid}, {"type", "message"}, {"role", "assistant"}, {"model", config.model_name},
                            {"content", json::array()}, {"stop_reason", nullptr}, {"stop_sequence", nullptr}, {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}};
                        write(sink, "event: message_start\n" + sse({{"type", "message_start"}, {"message", message}}));
                        if (has_tools) {
                            auto result = engine.generate(prompt, params);
                            auto parsed = parse_output(result.text);
                            size_t index = 0;
                            if (!parsed.text.empty()) {
                                write(sink, "event: content_block_start\n" + sse({{"type", "content_block_start"}, {"index", index}, {"content_block", {{"type", "text"}, {"text", ""}}}}));
                                write(sink, "event: content_block_delta\n" + sse({{"type", "content_block_delta"}, {"index", index}, {"delta", {{"type", "text_delta"}, {"text", parsed.text}}}}));
                                write(sink, "event: content_block_stop\n" + sse({{"type", "content_block_stop"}, {"index", index++}}));
                            }
                            for (const auto & call : parsed.calls) {
                                write(sink, "event: content_block_start\n" + sse({{"type", "content_block_start"}, {"index", index}, {"content_block", {{"type", "tool_use"}, {"id", call["id"]}, {"name", call["function"]["name"]}, {"input", json::object()}}}}));
                                write(sink, "event: content_block_delta\n" + sse({{"type", "content_block_delta"}, {"index", index}, {"delta", {{"type", "input_json_delta"}, {"partial_json", call["function"]["arguments"]}}}}));
                                write(sink, "event: content_block_stop\n" + sse({{"type", "content_block_stop"}, {"index", index++}}));
                            }
                            write(sink, "event: message_delta\n" + sse({{"type", "message_delta"}, {"delta", {{"stop_reason", parsed.calls.empty() ? "end_turn" : "tool_use"}, {"stop_sequence", nullptr}}}, {"usage", {{"output_tokens", result.stats.generated_tokens}}}}));
                            write(sink, "event: message_stop\n" + sse({{"type", "message_stop"}}));
                            sink.done(); return true;
                        }
                        write(sink, "event: content_block_start\n" + sse({{"type", "content_block_start"}, {"index", 0}, {"content_block", {{"type", "text"}, {"text", ""}}}}));
                        VisibleStream visible([&](const std::string & delta) {
                            return write(sink, "event: content_block_delta\n" + sse({{"type", "content_block_delta"}, {"index", 0}, {"delta", {{"type", "text_delta"}, {"text", delta}}}}));
                        });
                        auto result = engine.generate(prompt, params, [&](const std::string & delta) { return visible.feed(delta); });
                        visible.finish();
                        write(sink, "event: content_block_stop\n" + sse({{"type", "content_block_stop"}, {"index", 0}}));
                        write(sink, "event: message_delta\n" + sse({{"type", "message_delta"}, {"delta", {{"stop_reason", "end_turn"}, {"stop_sequence", nullptr}}}, {"usage", {{"output_tokens", result.stats.generated_tokens}}}}));
                        write(sink, "event: message_stop\n" + sse({{"type", "message_stop"}}));
                    } catch (const std::exception & e) { write(sink, "event: error\n" + sse({{"type", "error"}, {"error", {{"type", "api_error"}, {"message", e.what()}}}})); }
                    sink.done(); return true;
                });
                return;
            }
            auto result = engine.generate(prompt, params);
            auto parsed = parse_output(result.text);
            json content = json::array();
            if (!parsed.text.empty()) content.push_back({{"type", "text"}, {"text", parsed.text}});
            for (const auto & call : parsed.calls) {
                const std::string args = call["function"]["arguments"].get<std::string>();
                json input = json::parse(args, nullptr, false);
                if (input.is_discarded()) input = {{"_raw", args}};
                content.push_back({{"type", "tool_use"}, {"id", call["id"]},
                    {"name", call["function"]["name"]}, {"input", input}});
            }
            res.set_content(json{{"id", mid}, {"type", "message"}, {"role", "assistant"}, {"model", config.model_name},
                {"content", content}, {"stop_reason", parsed.calls.empty() ? "end_turn" : "tool_use"}, {"stop_sequence", nullptr},
                {"usage", {{"input_tokens", result.stats.prompt_tokens}, {"output_tokens", result.stats.generated_tokens},
                           {"cache_read_input_tokens", result.stats.cached_tokens}}}}.dump(), "application/json");
        } catch (const std::exception & e) { res.status = 400; res.set_content(json{{"type", "error"}, {"error", {{"type", "invalid_request_error"}, {"message", e.what()}}}}.dump(), "application/json"); }
    });

    server.set_error_handler([](const auto &, auto & res) {
        if (res.body.empty()) res.set_content(api_error("route not found", "not_found_error").dump(), "application/json");
    });
    server.set_exception_handler([](const auto &, auto & res, std::exception_ptr ep) {
        try { std::rethrow_exception(ep); } catch (const std::exception & e) {
            res.status = 500; res.set_content(api_error(e.what(), "server_error").dump(), "application/json");
        }
    });

    std::cerr << "listening on http://" << config.host << ':' << config.port << "\n";
    if (!server.listen(config.host, config.port)) throw std::runtime_error("cannot bind server socket");
    return 0;
}

} // namespace neutron
