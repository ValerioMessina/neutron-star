cmake_minimum_required(VERSION 3.24)
project(neutron_star VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(NS_CUDA "Build llama.cpp CUDA backend" OFF)
option(NS_NATIVE "Tune CPU helper code for this machine" ON)

include(FetchContent)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(APPLE)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
else()
  set(GGML_METAL OFF CACHE BOOL "" FORCE)
endif()
set(GGML_CUDA ${NS_CUDA} CACHE BOOL "" FORCE)
set(GGML_NATIVE ${NS_NATIVE} CACHE BOOL "" FORCE)
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
  src/llama_backend.cpp)
target_include_directories(neutron_core PUBLIC include)
target_link_libraries(neutron_core PUBLIC llama nlohmann_json::nlohmann_json)
target_compile_options(neutron_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neutron-star src/main.cpp src/server.cpp)
target_link_libraries(neutron-star PRIVATE neutron_core httplib::httplib)

enable_testing()
add_executable(neutron-tests tests/tests.cpp src/gemma4_prompt.cpp src/config.cpp)
target_include_directories(neutron-tests PRIVATE include)
target_link_libraries(neutron-tests PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME neutron-tests COMMAND neutron-tests)

install(TARGETS neutron-star RUNTIME DESTINATION bin)
