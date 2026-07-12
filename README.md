# Neutron Star

Neutron Star is a focused local inference service for **Gemma 4 12B**. The production path is Apple Silicon/Metal; the compute backend boundary is portable to CUDA. It provides incremental prefill, live token-prefix KV reuse, optional disk checkpoints, sampling, cancellation, and OpenAI/Anthropic-compatible HTTP APIs.

This is not a claim that a first implementation is already faster than every mature runtime. The current compute layer deliberately pins a known Gemma 4-capable `llama.cpp` commit, while Neutron Star owns model-specific prompt rendering, cache lifecycle and protocol behavior. Performance changes should only be accepted with repeatable benchmarks and output-regression tests.

## Tested machine and model

- Apple M3 Pro, 18 GB unified memory, macOS 14.3.1
- local Ollama `gemma4:12b`, Q4_K_M, 11.9B parameters, 6.86 GiB
- 48 transformer blocks, 262,144-token trained context
- hybrid attention: 40 sliding-window layers and 8 global layers
- all 49 loadable layers offloaded to Metal, Flash Attention enabled

The default runtime context is 8,192 tokens. On this 18 GB Mac, allocating the model's complete 262K KV context is not realistic. Increase `--ctx` only after measuring memory pressure.

## Build and run

Requirements: CMake 3.24+, Xcode command-line tools and Git.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure

./build/neutron-star \
  --model "$HOME/.ollama/models/blobs/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606" \
  --ctx 8192 \
  --kv-cache-dir "$HOME/.cache/neutron-star/kv"
```

The exact local `gemma4:12b` path is the default on the machine for which this repository was created. For another quant, pass `--model`. Use `--verbose` to show backend allocation and Metal kernel logs.

CUDA configuration is intentionally a build-time backend choice:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNS_CUDA=ON
cmake --build build-cuda -j
```

## API

Available routes:

- `GET /healthz`
- `GET /v1/models` and `GET /v1/models/{id}`
- `POST /v1/chat/completions` (OpenAI JSON and SSE)
- `POST /v1/completions`
- `POST /v1/responses` (OpenAI JSON and SSE event lifecycle)
- `POST /v1/messages` (Anthropic JSON and SSE event lifecycle)

OpenAI example:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gemma-4-12b",
    "messages":[{"role":"user","content":"Explain Redis streams briefly."}],
    "stream":true,
    "max_tokens":256
  }'
```

Anthropic example:

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model":"gemma-4-12b",
    "max_tokens":256,
    "messages":[{"role":"user","content":"Explain Redis streams briefly."}]
  }'
```

Set `NEUTRON_API_KEY` or pass `--api-key` to require either `Authorization: Bearer ...` or Anthropic's `x-api-key`. The default bind address is loopback only.

## Cache and execution design

One model context is a single mutable source of truth. HTTP parsing is concurrent, while graph execution is serialized under the engine lock. A request is rendered and tokenized, then compared with the live token history:

1. the common prefix remains in Metal KV buffers;
2. the divergent suffix is removed;
3. only new prompt tokens are prefetched in configurable chunks;
4. generated tokens extend that same state;
5. with `--kv-cache-dir`, the active sequence is written through a temporary file and atomically renamed.

An exact prompt hit replays its final token because sequence checkpoints contain KV state but do not provide a safe reusable logits row. This trades one token of work for correct sampling. API usage reports `prompt_tokens_details.cached_tokens` (OpenAI) or `cache_read_input_tokens` (Anthropic).

Gemma 4 prompts use the official `<|turn>...<turn|>` controls, optional `<|think|>`, native tool declarations, tool calls and tool responses. Standard previous-turn thought text is not injected back into later prompts. Tool calls are mapped to OpenAI `tool_calls` and Anthropic `tool_use` blocks.

## Verification and benchmarking

```sh
./scripts/smoke.sh
python3 scripts/bench.py --tokens 128
```

On the tested machine, the first tiny request (including first-use Metal pipeline compilation) took about 11 seconds. Repeating the same 25-token prompt reused 24 tokens and completed in about **0.49 s**. Treat this as a smoke result, not a comparative benchmark: thermal state, context length, output length and kernel warm-up must be controlled before making speed claims.

## Current engineering boundaries

- Text inference is validated. The separate Gemma projector file is detected outside this engine but image/audio embedding execution is not yet wired, so multimodal placeholders alone are insufficient for real media input.
- The active graph is single-stream and does not continuously batch unrelated requests.
- Disk persistence currently keeps the active session checkpoint, not a size-bounded multi-session LRU.
- CUDA is structurally supported by the pinned backend and build flag, but this repository has only been executed on Metal so far.
- A credible “faster than DwarfStar/other engines” result requires an agreed prompt corpus, context frontiers, quant, warm-up policy and quality/logit gate. No unsupported superiority claim is made here.

These are the next optimization targets: paged multi-session KV, continuous batching, Gemma 4 MTP speculative decoding, projector integration, official-logit regression vectors, and profiling-led Gemma-specific Metal fusion.

## Sources and licenses

The project is MIT licensed. Its pinned dependencies retain their own licenses. Gemma model use remains subject to Google's model terms. Design references include DwarfStar and `llama.cpp`; no DwarfStar source is copied into this repository.
