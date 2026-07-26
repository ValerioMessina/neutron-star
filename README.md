# Neutron Star

Native Gemma 4 12B inference for Apple Silicon. Neutron Star owns GGUF
loading, tokenization, the transformer graph, quantized Metal kernels, KV
state, sampling, multimodal input, and OpenAI/Anthropic-compatible HTTP APIs.

## Build

Requirements: macOS, Apple Silicon, Xcode command-line tools, and CMake 3.24+.

```sh
cmake --preset metal-release
cmake --build --preset metal-release
ctest --preset metal-release
```

Convert a compatible Gemma 4 Q4_K_M GGUF once for the direct Metal FFN layout:

```sh
./build/neutron-convert-metal-ffn MODEL.gguf MODEL.nsmffn
```

Run the server:

```sh
./build/neutron-star \
  --model MODEL.gguf \
  --metal-ffn MODEL.nsmffn \
  --ctx 8192 \
  --batch 1024
```

The indexed disk KV cache and quality-oriented sparse attention in the 40
sliding-window layers are enabled by default. The 40 local FFNs use FP16
accumulation; the eight global layers retain FP32 accumulation. Use
`--no-kv-cache`, `--dense-swa`, or `--exact-ffn` for conservative operation.

On the tested 18 GB M3 Pro, warm uncached prefill is about 177 token/s and the
five-position retrieval probe scores 5/5. An experimental 300+ token/s FFN
path was rejected because it scored 0/5; Neutron Star does not report that
mode as usable. Cached prefixes can be restored much faster than recomputing
them.

See [DOCUMENTATION.md](DOCUMENTATION.md) for conversion details, APIs,
benchmarks, multimodal support, cache behavior, and tuning.

Inspired by the long-prefill work in [DwarfStar](https://github.com/antirez/ds4)
and built with selected components from
[llama.cpp](https://github.com/ggml-org/llama.cpp).
