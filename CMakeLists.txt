cmake_minimum_required(VERSION 3.24)
project(neutron_star VERSION 0.1.0 LANGUAGES C CXX)

if(NOT APPLE)
  message(FATAL_ERROR "This revision provides the native Apple Metal backend only; CUDA is the next backend target")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_library(neutron_native_core src/native/gguf.cpp src/native/tokenizer.cpp src/native/quant.cpp src/native/model.cpp src/native/metal_ffn.cpp src/native/kv_cache.cpp)
target_include_directories(neutron_native_core PUBLIC include)
target_compile_options(neutron_native_core PRIVATE -Wall -Wextra -Wpedantic)
add_executable(neutron-native-inspect src/native/inspect.cpp)
target_link_libraries(neutron-native-inspect PRIVATE neutron_native_core)
add_executable(neutron-convert-metal-ffn tools/convert_metal_ffn.cpp)
target_link_libraries(neutron-convert-metal-ffn PRIVATE neutron_native_core)
if(APPLE)
  enable_language(OBJC OBJCXX)
  find_program(NEUTRON_METAL_COMPILER
    NAMES metal
    HINTS "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin"
    REQUIRED)
  find_program(NEUTRON_METALLIB_TOOL
    NAMES metallib
    HINTS "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin"
    REQUIRED)
  set(NS_METAL_AIR ${CMAKE_BINARY_DIR}/gemma4.air)
  set(NS_METALLIB ${CMAKE_BINARY_DIR}/gemma4.metallib)
  add_custom_command(OUTPUT ${NS_METALLIB}
    COMMAND ${NEUTRON_METAL_COMPILER} -O3 -c ${CMAKE_SOURCE_DIR}/metal/gemma4.metal -o ${NS_METAL_AIR}
    COMMAND ${NEUTRON_METALLIB_TOOL} ${NS_METAL_AIR} -o ${NS_METALLIB}
    DEPENDS metal/gemma4.metal VERBATIM)
  add_custom_target(neutron-metallib DEPENDS ${NS_METALLIB})
endif()

include(FetchContent)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
FetchContent_Declare(llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG e3546c7948e3af463d0b401e6421d5a4c2faf565
  GIT_SHALLOW TRUE)
FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_Declare(cpp_httplib URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.31.0.tar.gz)
FetchContent_MakeAvailable(llama_cpp nlohmann_json cpp_httplib)

add_library(neutron_core
  src/config.cpp
  src/gemma4_prompt.cpp
  src/native/multimodal.cpp
  src/native/metal_engine.mm)
target_include_directories(neutron_core PUBLIC include)
target_link_libraries(neutron_core PUBLIC neutron_native_core nlohmann_json::nlohmann_json mtmd llama "-framework Metal" "-framework Foundation")
add_dependencies(neutron_core neutron-metallib)
target_compile_definitions(neutron_core PRIVATE NEUTRON_METALLIB_PATH="${NS_METALLIB}")
target_compile_options(neutron_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neutron-star src/main.cpp src/server.cpp)
target_link_libraries(neutron-star PRIVATE neutron_core httplib::httplib)

add_executable(neutron-prefill-probe tools/prefill_probe.cpp)
target_link_libraries(neutron-prefill-probe PRIVATE neutron_core)

add_executable(neutron-decode-probe tools/decode_probe.cpp)
target_link_libraries(neutron-decode-probe PRIVATE neutron_core)

add_executable(neutron-sparse-accuracy-probe tools/sparse_accuracy_probe.cpp)
target_link_libraries(neutron-sparse-accuracy-probe PRIVATE neutron_core)

enable_testing()
add_executable(neutron-tests tests/tests.cpp src/gemma4_prompt.cpp src/config.cpp)
target_include_directories(neutron-tests PRIVATE include)
target_link_libraries(neutron-tests PRIVATE neutron_native_core nlohmann_json::nlohmann_json)
add_test(NAME neutron-tests COMMAND neutron-tests)

install(TARGETS neutron-star RUNTIME DESTINATION bin)
