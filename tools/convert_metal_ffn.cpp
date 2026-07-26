#include "neutron/native/gguf.hpp"
#include "neutron/native/metal_ffn.hpp"
#include "neutron/native/model.hpp"

#include <iostream>

int main(int argc, char ** argv) {
    const bool in_place = argc > 1 && std::string(argv[1]) == "--in-place";
    if ((in_place && argc != 4) || (!in_place && argc != 3)) {
        std::cerr << "usage: neutron-convert-metal-ffn MODEL.gguf OUTPUT.nsmffn\n"
                     "       neutron-convert-metal-ffn --in-place MODEL.gguf METADATA.gguf\n";
        return 2;
    }
    try {
        const char * model_path = argv[in_place ? 2 : 1];
        const char * output_path = argv[in_place ? 3 : 2];
        neutron::native::GGUF gguf(model_path, !in_place);
        neutron::native::Gemma4Model model(gguf);
        if (in_place) {
            neutron::native::convert_metal_ffn_in_place(model_path, output_path, gguf, model);
            neutron::native::GGUF metadata(output_path, false);
            neutron::native::Gemma4Model metadata_model(metadata);
            neutron::native::MetalFfnFile verified(model_path, metadata, metadata_model);
            std::cout << "created=" << model_path << " metadata=" << output_path
                      << " bytes=" << verified.file_size() << "\n";
        } else {
            neutron::native::convert_metal_ffn(output_path, gguf, model);
            neutron::native::MetalFfnFile verified(output_path, gguf, model);
            std::cout << "created=" << output_path << " bytes=" << verified.file_size() << "\n";
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
