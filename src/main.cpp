#include "neutron/config.hpp"
#include "neutron/engine.hpp"
#include "neutron/server.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        auto config = neutron::Config::from_args(argc, argv);
        if (!std::filesystem::is_regular_file(config.model_path)) {
            throw std::runtime_error("model file not found: " + config.model_path);
        }
        std::cerr << "loading " << config.model_path << "\n";
        auto engine = neutron::make_native_engine(config);
        std::cerr << "model: " << engine->model_description() << "\n"
                  << "context: " << engine->context_size() << " tokens\n";
        return neutron::run_server(config, *engine);
    } catch (const std::exception & e) {
        if (std::string(e.what()) == "help") {
            std::cout << neutron::Config::usage(argv[0]);
            return 0;
        }
        std::cerr << "fatal: " << e.what() << "\n\n" << neutron::Config::usage(argv[0]);
        return 1;
    }
}
