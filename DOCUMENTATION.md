# Neutron Star — Full Documentation

Neutron Star is an independent inference runtime specialized for **Gemma 4 12B Q4_K_M on Apple Silicon**. The production executable does not link or call `llama.cpp`: it owns GGUF loading, Gemma tokenization, the transformer graph, quantized Metal kernels, KV state, sampling, prompt rendering, and the HTTP APIs.

## Native execution path

- zero-copy, mmap-backed GGUF v3 weights;
- native Q4_K and Q6_K decode kernels;
- greedy Gemma 4 Assistant MTP speculative decoding with one-to-three-token target verification;
- 64×32 swizzled Q4_K/Q6_K prefill tiles with paired 64-wide K dequantization and Metal `simdgroup_matrix` accumulation;
- aligned 128-bit Q4_K payload loads, paired-lane Q4 scale/min decoding exchanged with `simd_shuffle_xor`, and packed 16-bit Q6_K extraction that decodes four values per mask/shift group directly into the quantized matrix tile;
- Gemma-specific Q4_K `gate + up + GEGLU` prefill; the direct-Metal model uses a fused 32-row K64 tile that assigns two SIMD groups to each projection and keeps the FP32 gate/up intermediates inside the threadgroup;
- direct `.nsmffn` Metal model layout: every tensor keeps a stable logical offset while gate/up/down blocks are stored as `[64-row slab][K block][row]`, so the hot FFN path consumes its native K-major order without startup repacking;
- 1,024-token prefill chunks by default; the direct FP16 attention path now covers every supported batch through 2,048 instead of falling back above 768, eliminating a nearly empty second model pass for prompts just over the old boundary;
- batch-adaptive prefill attention without a global score tensor: the eight global layers use a 16-query/64-key Metal schedule through a 1,024-token span and the register-resident 8-query/128-key long-context schedule thereafter, while the 40 sliding-window layers retain their faster 32-query specialization;
- fused post-projection stages: Q RMSNorm + RoPE in one dispatch, K/V RMSNorm + K RoPE + KV-ring storage in another, and residual-add + next RMSNorm + FP16 conversion without a second readback;
- two-lane cooperative QK attention for decode; a two-row-per-SIMD-group Q4_K `gate + up + GEGLU` kernel shares activation loads and keeps only the FFN intermediate;
- asynchronous `MADV_WILLNEED` SSD prefetch for the zero-copy GGUF mapping, plus Metal residency sets on macOS 15 and newer; hot inference does not stream a model that already fits unified memory;
- all 48 Gemma 4 blocks, including 40 sliding-window and 8 global-attention layers;
- head RMS normalization, NeoX RoPE with global frequency factors, GEGLU, soft caps, and tied output embeddings;
- private FP16 K/V buffers per layer, live longest-prefix reuse, and optional
  content-addressed disk snapshots with indexed longest-prefix lookup;
- greedy, temperature, top-k, top-p and min-p sampling;
- OpenAI chat/completions/responses and Anthropic messages endpoints, including SSE and tool-call protocol conversion.

The compute boundary is isolated under `src/native/` and `metal/`. A future CUDA implementation can implement the same `Engine` interface without changing the server, prompt, tokenizer, or API layers. CUDA is not implemented yet.

## Tested target

- Apple M3 Pro, 18 GB unified memory;
- local Ollama `gemma4:12b` GGUF, Q4_K_M, about 6.86 GiB;
- 48 layers, embedding width 3840, trained context 262,144;
- trained and runtime-validated context limit 262,144 tokens; the default CLI allocation remains 8,192 to avoid reserving the full KV cache accidentally.

## Build and run

Requirements: CMake 3.24+ and Xcode command-line tools.

```sh
cmake --preset metal-release
cmake --build --preset metal-release
ctest --preset metal-release

./build/neutron-star \
  --model "$HOME/Library/Caches/neutron-star/models/gemma4-12b-down-q4-metal-metadata.gguf" \
  --metal-ffn "$HOME/Library/Caches/neutron-star/models/gemma4-12b-down-q4-metal.nsmffn" \
  --mmproj "$HOME/.ollama/models/blobs/sha256-675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842" \
  --draft-model "$HOME/Library/Caches/neutron-star/models/gemma-4-12B-it-assistant-MTP-Q4_K_M.gguf" \
  --ctx 262144 \
  --batch 1024
```

The original Ollama GGUF can still be auto-discovered on the development machine. The direct-Metal pair must be selected explicitly with `--model` plus `--metal-ffn`, as shown above. The loader fingerprints the pair and rejects an incompatible tensor layout instead of silently executing the wrong graph.

## Direct Metal conversion and selective quantization

The deployed model was produced directly from the original QAT-derived GGUF in two bounded-memory stages. The first stage requantizes the heavy Q6_K down projections to Q4_K while preserving the tied embedding and attention V projections as Q6_K. Gate/up were already Q4_K and remain quantized. The second stage converts the whole working GGUF in place to the Metal layout and creates a sparse GGUF metadata companion:

```sh
llama-quantize --allow-requantize \
  --token-embedding-type q6_k \
  --tensor-type 'ffn_down=q4_k' \
  --tensor-type 'attn_v=q6_k' \
  SOURCE.gguf METAL-WORKING.gguf Q4_K_S 6

./build/neutron-convert-metal-ffn --in-place \
  METAL-WORKING.gguf METAL-WORKING-metadata.gguf
```

`--in-place` deliberately makes `METAL-WORKING.gguf` no longer usable as an ordinary GGUF, so the source must be retained. It reorders one sub-megabyte FFN slab at a time, writes the `NSMFFN03` header only after all tensors complete, and leaves non-FFN tensors at their original offsets. The sparse metadata file has the original logical size but occupies about 15 MiB physically.

Gate and up are already Q4_K in the source model. A tested Q2_K overlay did
not improve kernel time on M3 Pro and introduced measurable error, so it is
not part of the runtime. The production path instead uses FP16 accumulation
only in the 40 sliding-window FFNs and retains FP32 accumulation in the eight
global layers. Pass `--exact-ffn` to use FP32 accumulation everywhere.

## APIs

Routes:

- `GET /healthz`
- `GET /v1/models` and `GET /v1/models/{id}`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/messages`

Both OpenAI and Anthropic streaming lifecycles are supported. Set `NEUTRON_API_KEY` or use `--api-key` to require `Authorization: Bearer ...` or `x-api-key`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gemma-4-12b",
    "messages":[{"role":"user","content":"Reply exactly: pong"}],
    "temperature":0,
    "max_tokens":16
  }'
```

### Vision, audio, and video

Pass the matching Gemma 4 `mmproj` GGUF with `--mmproj` (or
`NEUTRON_MMPROJ`) to enable the three media modalities. The text transformer
and KV cache remain in Neutron's native Metal engine; the pinned llama.cpp
`mtmd` encoder supplies the projected 3840-wide media embeddings. Vision uses
non-causal attention within each projected image, while audio and subsequent
text retain causal attention. Video is decoded to visual frames through
`ffmpeg`/`ffprobe`, which must be available on `PATH`.

The OpenAI Chat Completions and Responses shapes (`image_url`, `input_image`,
`input_audio`, `input_video`, and `video_url`) and Anthropic base64 `source`
objects are accepted. Media may be a base64 data URL or a local path. Remote
HTTP media URLs are deliberately rejected; fetch them in the client and send a
data URL instead. The server payload limit is 256 MiB.

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe the image."}
    ]
  }]
}
```

## OpenClaw default engine

OpenClaw 2026.7.1 is installed under `~/.openclaw` with `neutron/gemma-4-12b` as the default model. Its custom `openai-completions` provider points to `http://127.0.0.1:8080/v1`, advertises 262,144 context tokens, and uses `models.providers.neutron.localService` to start this repository's server on demand with the direct-Metal model and Gemma Assistant MTP. The local profile disables workspace/skill injection and exposes only the minimal tool profile so the agent harness does not consume the small-model prompt budget unnecessarily. See OpenClaw's official [custom provider configuration](https://docs.openclaw.ai/gateway/config-tools) and [local model service](https://docs.openclaw.ai/gateway/local-model-services) documentation.

The validated smoke command is:

```sh
~/.openclaw/bin/openclaw config validate
~/.openclaw/bin/openclaw agent --local --session-id neutron-smoke \
  --message 'Reply exactly: local engine active' --thinking off --json
```

The smoke test confirmed provider autostart, a 262,144-token agent budget, a successful streamed Neutron response, and clean child-process shutdown.

## KV reuse

The engine keeps one mutable in-memory sequence. Each request is tokenized and compared with the current sequence. Matching tokens retain their K/V data; only the divergent suffix is recomputed. An exact prompt hit deliberately replays its last token to recreate a valid logits row. Cache use is reported as OpenAI `prompt_tokens_details.cached_tokens` and Anthropic `cache_read_input_tokens`.

For reuse across unrelated requests and process restarts, the indexed disk
cache is enabled by default. Without an override it uses
`$HOME/Library/Caches/neutron-star/kv`; pass `--no-kv-cache` (or set
`NEUTRON_KV_CACHE=0`) to disable it:

```sh
./build/neutron-star \
  --model /path/to/model.gguf \
  --metal-ffn /path/to/model.nsmffn \
  --kv-cache-entries 8 \
  --kv-cache-min-tokens 256
```

Snapshots are content-addressed by model fingerprint, token count, and a
rolling token hash. Startup indexes only their filenames; lookup computes all
prompt-prefix hashes once and validates the winning file's complete token list
before mapping it. The snapshot records every native layer dimension, ring
capacity, and active slot count, so a context, batch, or model mismatch is a
cache miss rather than a corrupt restore. Sparse-attention settings are mixed
into the model fingerprint so dense and approximate KV states cannot be
cross-restored. Writes use a temporary file,
`fsync`, and atomic rename, and the oldest files are pruned to the configured
limit. Multimodal prompts are deliberately excluded because their projected
embeddings are not represented by text token IDs.

In the local 225-token restart test, cold prefill took 1,302.86 ms. Restoring
224 tokens from a 74 MiB snapshot and replaying the final token took 64.16 ms,
a 20.3x prefill-latency reduction, with the same generated-text hash. Snapshot
size grows with the actual prefix (sliding layers remain ring-bounded); keep
the entry limit small on space-constrained machines. Environment equivalents
are `NEUTRON_KV_CACHE`, `NEUTRON_KV_CACHE_DIR`,
`NEUTRON_KV_CACHE_ENTRIES`, and `NEUTRON_KV_CACHE_MIN_TOKENS`.

### Sparse attention in the 40 sliding-window layers

The direct Metal prefill path approximates all 40 sliding-window attention
layers by retaining the oldest and newest 64-token blocks and sampling the
blocks between them. The quality-oriented stride 2 profile is enabled by
default:

```sh
./build/neutron-star -m /path/to/model.gguf \
  --sparse-swa-threshold 128 \
  --sparse-swa-sink 64 \
  --sparse-swa-recent 64 \
  --sparse-swa-stride 2
```

The default stride 2 is the quality-oriented preset. On the local M3 Pro,
the median 2,205-token prefill improved from 164.83 to 168.48 token/s
(2.2%), while the five-position retrieval probe remained 5/5. Stride 16
reached 172.07 token/s (4.4%) but fell to 2/5 on that probe, so aggressive
strides remain available but are not the default. At 225 tokens the measured
rate stayed essentially unchanged at about 170 token/s.

Use `--dense-swa` to restore exact sliding-window attention.

This optimization cannot produce 300 token/s on the tested machine: the FFN
gate/up and down projections dominate prefill, and even removing attention
entirely would leave their cost. Reused prefixes do exceed that effective
rate through the default disk KV cache; uncached prompts do not. Environment
equivalents are `NEUTRON_SPARSE_SWA`, `NEUTRON_SPARSE_SWA_THRESHOLD`,
`NEUTRON_SPARSE_SWA_SINK`, `NEUTRON_SPARSE_SWA_RECENT`, and
`NEUTRON_SPARSE_SWA_STRIDE`.

## Verification and benchmarks

```sh
./build/neutron-native-inspect /path/to/model.gguf
./build/neutron-prefill-probe /path/to/model.gguf 768 200
./build/neutron-sparse-accuracy-probe /path/to/model.gguf swa
./scripts/smoke.sh
python3 scripts/bench.py --tokens 128
```

### Opt-in sparse long context

Neutron Star has an approximate sparse path for causal global attention. It is
disabled by default, so the normal path keeps dense llama.cpp-style semantics.
Enable it explicitly:

```sh
./build/neutron-star -m /path/to/model.gguf -c 262144 \
  --sparse-context \
  --sparse-threshold 65536 \
  --sparse-sink 1024 \
  --sparse-window 32768 \
  --sparse-stride 8
```

The schedule always retains the prefix sink and recent window, and keeps one
old 128-token block per stride in the middle. It applies only to causal global
attention; sliding-window and non-causal multimodal attention are unchanged.
Environment equivalents are `NEUTRON_SPARSE_CONTEXT=1`,
`NEUTRON_SPARSE_THRESHOLD`, `NEUTRON_SPARSE_SINK`,
`NEUTRON_SPARSE_WINDOW`, and `NEUTRON_SPARSE_STRIDE`.

This mode is not bit-exact and cannot guarantee the same answer as dense
llama.cpp. The included `neutron-sparse-accuracy-probe` makes the tradeoff
measurable. In a deliberately extreme 2,842-token test (256-token sink,
512-token recent window, stride 16), dense retrieval scored 5/5 while sparse
retrieval scored 3/5: a 40 percentage-point loss. Use the larger defaults above
for real long contexts and validate them on the target workload. The same
extreme setup processed a cold 9,905-token probe at 152.12 token/s versus the
dense probe's 139.68 token/s, an 8.9% gain at that length.

Measured on the M3 Pro after pipeline warm-up:

- native 258-token prompt: about **170.8 token/s**;
- native 511-token prompt: about **169.1 token/s**;
- native 1,105-token prompt with 768-token chunks: **162.5 token/s**;
- native 2,205-token prompt with 768-token chunks: **158.1 token/s**;
- native 128-token greedy decode: **16.37 token/s**;
- Q6_K down projection: about **3.62 effective TFLOPS** at batch 256 and **3.83 effective TFLOPS** at batch 1024;
- exact-prefix request: 12 of 13 prompt tokens reused.

The end-to-end measurements were run locally and sequentially as the median of three divergent warm prompts; reused prefix tokens are excluded from the throughput numerator. Model residency, thermal state, context settings, and output length can change the result. First-use pipeline compilation is excluded from warm figures. A freshly compiled local `llama.cpp` (`e3546c794`, Metal flash attention, batch/ubatch 768) measured 155.0, 172.2, 168.2, and 167.8 token/s at 258, 511, 1,105, and 2,205 prompt tokens, respectively, plus 16.95 token/s for its decode benchmark. The default Neutron path at the time was still about 6% behind at 2,205 tokens; the then-opt-in concurrent gate/up path narrowed that gap to about 4.6%, was later made automatic for the direct-Metal model, and has now been superseded by the fused RM32 default described below. Its decode probe also includes greedy sampling, so that row is not a pure evaluator-only comparison.

A final rerun on 2026-07-15 used the direct-Metal selectively quantized model for Neutron, the original Q4_K_M GGUF for `llama.cpp e3546c794`, batch/ubatch 768, Metal flash attention, and three warmed repetitions:

| Runtime | 225-token prefill | 775-token prefill | 2,205-token prefill | 64-token generation |
| --- | ---: | ---: | ---: | ---: |
| Neutron target only | **175.2 tok/s** | 161.1 tok/s | 158.3 tok/s | **17.32 tok/s** |
| Neutron + 0.4B Gemma Assistant MTP | same target prefill | same target prefill | same target prefill | **21.96 tok/s** |
| llama.cpp `e3546c794` | 152.1 tok/s | **161.2 tok/s** | **168.3 tok/s** | 16.88 tok/s |
| MLX-VLM 0.6.4 / MLX 0.32.0 | 132.2 tok/s | **163.7 tok/s** | 166.9 tok/s | 11.95 tok/s |

The MLX row is a fresh rerun with the official 10 GB affine checkpoint. MLX generation is ordinary autoregressive decoding without the Gemma Assistant used by Neutron. Representations remain different: Neutron uses selectively quantized Q4_K/Q6_K weights, `llama.cpp` uses the original Q4_K_M GGUF, and MLX uses its checkpoint's per-layer affine 4/8-bit scheme.

The 2026-07-25 long-prefill change removed the artificial 768-query cutoff
from the direct FP16 attention kernels and raised Neutron's default chunk to
1,024. A same-session rerun on the M3 Pro used three warmed repetitions per
length, the same original Q4_K_M values (reordered losslessly for Neutron), and
Metal flash attention. The table reports the median of each engine's three
samples:

| Runtime | 225-token prefill | 775-token prefill | 2,205-token prefill |
| --- | ---: | ---: | ---: |
| Neutron, batch 1,024 | **170.57 tok/s** | **163.21 tok/s** | **164.65 tok/s** |
| llama.cpp `e3546c794`, batch/ubatch 768 | 152.11 tok/s | 162.17 tok/s | 156.39 tok/s |
| Neutron advantage | **12.1%** | **0.6%** | **5.3%** |

The llama.cpp 2,205-token samples ranged from 156.36 to 167.20 tok/s as the
device heated, while Neutron's corresponding samples stayed between 164.29
and 164.81 tok/s. These are reproducible results for this machine and suite,
not a hardware-independent guarantee.

Configured-context memory was measured separately with one minimal inference and `/usr/bin/time -l`. This isolates KV allocation growth and startup-touched model pages; it is not the fully resident memory of a long prefill:

| Runtime | 8,192 context | 65,536 context | 131,072 context | 262,144 context |
| --- | ---: | ---: | ---: | ---: |
| Neutron direct Metal, target only | 1.13 GB | 2.07 GB | not repeated | 5.29 GB |
| Neutron direct Metal + MTP | not repeated | not repeated | not repeated | 5.31 GB |
| llama.cpp Q4_K_M | 0.82 GB | 1.76 GB | not repeated | 5.04 GB |
| MLX active model + materialized KV | 11.46 GB | 12.40 GB | 13.47 GB | unsupported by checkpoint |

Both GGUF engines allocated the full 262,144-token KV configuration without swap on the 18 GB M3 Pro. The downloaded MLX checkpoint declares `max_position_embeddings: 131072`; its full limit uses 2.483 GB for BF16 KV and measured 13.473 GB of active MLX memory, with a 13.921 GB process peak. These footprint rows are not perfectly symmetric: file-backed GGUF pages remain mmap-backed, while MLX materializes its model into private arrays counted in active memory. Neutron also completed a real prompt above 8,000 tokens. Sliding-window layers retain their bounded ring, while global decode now uses the same register-resident online-softmax strategy as long prefill instead of materializing the complete score vector.

The long-prefill global-attention path now follows llama.cpp's tiled online
softmax schedule: eight queries share a 128-key tile, keep the 8x512 value
accumulator in SIMD registers, and use a 16 KiB threadgroup allocation. It is
selected automatically once a global span exceeds 1,024 tokens. A cold
9,905-token correctness probe at a 16,384-token configured context completed
in 70,912.7 ms (139.68 token/s); this is a correctness measurement including
first-use costs, not a warmed cross-engine benchmark.

Generation after a long prompt uses a fused global-attention path once the
span exceeds 2,048 tokens. It keeps softmax state and the 8x512 value
accumulator inside one Metal dispatch, writes FP16 attention directly, and
converts only the final 8,192 values needed by the output projection. The old
path issued separate QK, softmax, and PV dispatches and materialized
`heads x context` FP32 scores. Sequential eight-token probes with identical
greedy hashes measured:

| Prompt | Previous decode | Fused decode | Gain |
| ---: | ---: | ---: | ---: |
| 2,432 tokens | 12.54 tok/s | 14.48 tok/s | **15.5%** |
| 4,832 tokens | 10.69 tok/s | 14.21 tok/s | **32.9%** |
| 9,032 tokens | 8.97 tok/s | 12.20 tok/s | **36.0%** |

An additional two-run 12,632-token validation measured 12.59 and
12.68 token/s with the same greedy hash in both runs.
`NEUTRON_LEGACY_LONG_DECODE=1` retains the three-pass diagnostic fallback.
The fused kernel is used only when its complete 128-key read is inside the KV
allocation; the final near-capacity tile falls back to the bounds-checked path.

For historical comparison, the focused prefill rerun on 2026-07-14 used batch/prefill step 768, three warmed repetitions, and no generated-token timing:

| Runtime | Model representation | 225 tokens | 2,205 tokens |
| --- | --- | ---: | ---: |
| Neutron | Gemma 4 12B GGUF Q4_K_M, 6.86 GiB | **174.7 tok/s** | 158.1 tok/s |
| llama.cpp `e3546c794` | same GGUF, Metal FA | 151.9 tok/s | **168.2 tok/s** |
| MLX-VLM 0.6.4 / MLX 0.32.0 | Gemma 4 12B MLX affine 4-bit, about 10 GB | 132.4 tok/s | 167.4 tok/s |

The historical MLX row used the official
[`mlx-community/gemma-4-12B-4bit`](https://huggingface.co/mlx-community/gemma-4-12B-4bit)
checkpoint. It is the same base architecture but not the same quantized weight
representation, so it is an engine-level comparison rather than a
bit-identical model comparison.

Three prefill alternatives are implemented for controlled A/B testing:

- `NEUTRON_FUSED_QKV=1` selects one Q4_K/Q6_K/tied-V projection dispatch that loads each activation tile once and emits Q, K, and V. The existing fused post-projection dispatches then apply Q norm + RoPE and K/V norm + K RoPE + direct FP16 KV-cache storage.
- `NEUTRON_PACKED_FFN=1` selects a layer-local K-major packed Q4_K gate/up layout and the fused gate + up + GEGLU projection, avoiding the persistent 708 MB scale/min sidecar.
- `NEUTRON_PERSISTENT_PACKED_FFN=1` packs every gate/up matrix once at startup and reuses the K-major representation across requests. It also disables whole-GGUF Metal residency so macOS can evict the original gate/up pages instead of retaining both layouts.

The first two alternatives reproduced the default decode-probe greedy hash. They remain opt-in because end-to-end A/B testing did not justify enabling them: fused QKV reduced the original 2,205-token probe from 159.9 to 151.5 tok/s, while layer-local packed FFN measured 159.8 tok/s. The QKV kernel saves activation reads in principle, but its larger live tile/register footprint reduces occupancy on the tested M3 Pro; the packed FFN's per-layer repacking cost consumes its projection gain.

Persistent packed FFN was a positive graph result before the Q32 attention change: it measured **176.7 versus 174.3 token/s** at 225 tokens and **154.6 versus 152.5 token/s** at 2,205 tokens. Combined with Q32 attention it reached about 159.5 token/s. Its packed buffer is 3.30 GiB; retaining the whole mapped GGUF in a residency set at the same time caused compression and swap on the 18 GB test Mac, while allowing the unused original gate/up pages to be evicted kept three long runs stable. It remains opt-in: startup work and virtual allocation are larger, and the different accumulation order can change a greedy token when logits are nearly tied.

The default online sliding-window attention now uses a 32-query tile specialized for Gemma's 256-wide SWA heads. It keeps one output channel per thread and doubles query reuse without increasing the per-thread PV accumulator footprint. The same-output long-context median improved from 152.5 to **158.1 token/s**, while the 225-token result remained neutral at 174.7 token/s. Set `NEUTRON_SWA_QT16_ATTENTION=1` to restore the previous Q16 diagnostic path.

The eight 512-wide global-attention layers use a FlashAttention-4-inspired Metal 3 schedule: a 64-key tile keeps all eight SIMD groups active during QK, causal masking and online softmax remain in the same threadgroup, each V value is reused across 16 queries, and the kernel writes FP16 directly into the following projection input. The tiled kernel has no 768-query correctness limit and is now selected for every configured prefill batch through 2,048. This is a Metal-specific implementation, not the CUDA/Blackwell FA4 kernel. A sequential three-run A/B on 2026-07-19 measured 175.01 versus 174.96 token/s at 225 tokens, 160.77 versus 160.51 at 775, and **158.06 versus 157.47** at 2,205. The 64-token greedy hash stayed identical at batch 768 and batch 384. Set `NEUTRON_LEGACY_GLOBAL_ATTENTION=1` for the previous 32-key/FP32-output path. Applying the diagnostic 64-key schedule to the SWA layers or selecting the diagnostic 64-token quantized matmul tile did not improve end-to-end throughput, so neither is selected by the production runtime.

Above a 1,024-token global-attention span, the default starts from the
`llama.cpp` Metal execution shape that matters on the tested M3, then uses a
Neutron-specific 8-query/128-key tile. Direct K/V reads feed the
`simdgroup_matrix` Q×K and P×V stages, while the 8×512 FP32 output accumulator
stays in SIMD registers across cache tiles. Diagonal matrix rescaling preserves
online-softmax semantics without spilling that accumulator, and overlapping
query storage with final conversion scratch reduces threadgroup memory from
26 KiB to 16 KiB. The inactive Metal cooperative-tensor path was not copied:
`llama.cpp e3546c794` reports `has tensor = false` on M3 Pro.

At 2,205 tokens the refined path measured 159.25 token/s with the established
output hash. At 8,002 tokens it measured 149.43 token/s versus 128.94 token/s
for Neutron's previous global-attention kernel: a **15.9% end-to-end gain** with
the same output hash. The corresponding `llama-bench` result was 156.16
token/s; exceeding that result by 10% cannot be obtained from global attention
alone because global attention accounts for less than 2.78 seconds of the
53.54-second refined prefill.
Set `NEUTRON_LEGACY_GLOBAL_PV_ATTENTION=1` to restore Neutron's previous
16-query scalar-PV schedule.

A thermally sustained same-session rerun on 2026-07-26 used three repetitions
on the 14-core M3 Pro GPU. Neutron processed 8,189-token prompts with batch
2,048 at 156.20, 157.88, and 158.38 token/s. `llama-bench e3546c794` processed
8,192 tokens with batch/ubatch 768 at 154.93, 156.26, and 155.14 token/s.
The medians are therefore **157.88 token/s for Neutron** and **155.14 token/s
for llama.cpp**, a 1.8% Neutron advantage in this run. Profiling attributes
roughly 48% of Neutron's measured GPU stage time to fused gate/up and 25% to
the down projection; increasing the batch, changing attention alone, or using
the tested alternate tile geometries did not approach 200 token/s.

An Apple Neural Engine feasibility pass on the same machine moved a complete
real-shape FFN (`gate + up + GELU + multiply + down`) into a Core ML ML Program.
The [Core ML compute plan](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html)
selected `MLNeuralEngineComputeDevice` for every operation. A single INT8
per-channel layer measured 49.45 ms at batch 1,024 and 97.47 ms at batch 2,048
through Python; the native Core ML call measured 90.16 ms at batch 2,048. Its
batch-1,024 output had 1.30% normalized RMS error and 0.99991 cosine similarity
against the FP16 layer. Four-bit palettization was faster (33.58 ms at batch
1,024) but inaccurate for these Q4_K-derived weights (43.9% normalized RMS
error, 0.918 cosine). The single-layer result did not survive the 48-layer
runtime: 48 resident programs failed their first ANE dispatch, 32 resident ANE
layers plus 16 Metal layers achieved only 34.45 token/s at 2,205 tokens, and
loading one ANE program at a time achieved 67.95 token/s. Core ML weight/program
paging dominated the matrix speedup, so no ANE backend is enabled in production.
The rejected probes were removed from the production tree. This matches
Apple's guidance to benchmark compressed models on the target hardware because
decompression and device placement vary by backend in the
[Core ML optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).

The 32-query sliding-window attention specialization now also writes FP16 directly into the output-projection input, removing 40 FP32 output writes and conversion dispatches per chunk. On the M3 Pro/Tahoe build, a warmed three-run A/B against the previous FP32-output path measured **170.10 versus 169.82 token/s** at 225 tokens and **159.60 versus 159.32** at 775. Combining both execution orders at 2,205 tokens measured **154.81 versus 154.55 token/s**. Output hashes matched at every tested length. Set `NEUTRON_LEGACY_SWA_F32_ATTENTION=1` to retain the previous path for comparison.

The SWA kernel now derives its key base per 32-query tile instead of per
microbatch. Keys that are outside the sliding window for every query in a tile
are therefore never loaded or multiplied. Contiguous K blocks are read
directly by the SIMD matrix instructions; only the single block crossing the
ring-buffer boundary uses threadgroup staging. On the same M3 Pro, the warmed
2,205-token probe improved from **158.59 to 163.83 token/s** (+3.30%) with the
same generated-text hash. This narrows, but does not yet reverse, the measured
gap to the 168.3 token/s llama.cpp reference at that length.

The same SWA path also avoids the runtime integer remainder in every K/V ring
lookup. A block computes its physical slot once; subsequent keys use an
increment plus a single wrap comparison, which is exactly equivalent for the
KV layout. In a consecutive, thermally warm 2,205-token A/B this changed
**158.06 to 161.21 token/s** (+1.99%); a repeat reached **161.84 token/s**.
The generated-text hash remained `5987285406307608496`. These hot-run numbers
are kept separate from the earlier cold-run result because absolute Metal
throughput varied materially with device temperature.

The production SWA kernel now combines that per-tile base with a 64-key tile.
It retains the 32-query specialization but halves the online-softmax
synchronization rounds; the older generic K64 diagnostic did not trim its key
range and therefore did not show this benefit. In the final warmed A/B, the
2,205-token median moved from 164.34 to 164.65 tok/s and the 775-token median
from 162.98 to 163.21 tok/s, with matching trial-0 greedy hashes. Set
`NEUTRON_SWA_K32_ATTENTION=1` to restore the K32 path.

On Metal 4, multi-chunk prefill uses unretained command buffers and queues
intermediate chunks without a CPU wait. Token IDs occupy stable offsets in a
context-sized shared buffer, so the CPU can prepare the next chunk while the
GPU consumes the previous one; the final command buffer remains the single
synchronization point and also validates every deferred command. Set
`NEUTRON_LEGACY_PREFILL_SUBMISSION=1` to restore one CPU/GPU wait per chunk.
On the tested M3 Pro, an alternating six-run 2,205-token A/B measured 148.56
tok/s versus 148.34 tok/s by median (about +0.15%, within run-to-run noise);
225-token prefill is neutral because it fits in one chunk. This is therefore a
small submission/latency cleanup, not a GPU-compute throughput claim.

The direct-Metal model now defaults to a fused RM32×TM32×K64 gate/up tile: two SIMD groups compute gate and two compute up, all four retain one FP32 accumulator bank, and the GEGLU epilogue writes FP16 directly for down projection. The fused tile alone measured **159.97 versus 158.64 token/s** in a sequential three-run 2,205-token A/B against the previous two concurrent RM64 matmuls. Having the two lanes assigned to each Q4 row decode one scale/min pair each and exchange it with `simd_shuffle_xor` then raised the final median to **161.19 token/s**; the corresponding 225-token median was 178.67 versus 175.43 for the previous path. The final full 64-token run measured **161.31 token/s** with the same greedy hash. This path also removes about 90 MiB of FP32 gate/up workspace at batch 768 and still avoids the legacy 708 MB scale/min buffer. Set `NEUTRON_CONCURRENT_GATE_UP=1` for the previous direct-Metal path or to opt into concurrency on plain GGUF; `NEUTRON_SERIAL_GATE_UP=1` retains the serial A/B fallback. Applying the same concurrent scheduling to Q/K/V was slower and was removed.

Two wider direct-layout FFN experiments were rejected on the same machine: an RM128×TM32×K32 down tile preserved the hash but measured 170.02 versus 176.52 token/s on the short probe, while a K32 split gate/up tile measured 169.79 and changed the hash. Both were removed from the production runtime.

Metal 4 cooperative tensors, an RM32×TM64 tile, and a second SoA weight layout
were tested and then removed. The installed M3 Pro toolchain does not expose
the tensor API, while the wider tile and SoA layout were neutral or slower.
Keeping those paths would have increased maintenance cost without improving
the supported machine.

For a same-date reference, `llama-bench e3546c794` with Metal flash attention,
batch/ubatch 768 and the same original GGUF measured 172.18 token/s at 768
prompt tokens.  Neutron's warmed 775-token median was 159.40 token/s.  Thus the
requested 20% dense-FFN advantage was not reached on this GPU; enabling either
experimental path by default would make the measured result worse, not better.

Compiling the entire Metal library with uniform `-ffast-math` was also tested rather than inferred: 153.20 versus 153.21 token/s in a same-binary long-context A/B before the Q32 change, effectively neutral. A targeted fast `tanh` epilogue was likewise neutral, so precise activation math remains the default.

The 64-wide K kernels were retained after isolated Q4_K/Q6_K tests improved by about 2.7–2.9% with unchanged numerical error; end-to-end gains are smaller because attention, norms, and graph overhead remain. The fused two-row decode FFN measured 16.73 versus 16.58 token/s for the legacy two-GEMV path over 64 generated tokens, with identical output hashes. Larger four/eight-row variants, a specialized tied-K/V store, GPU argmax, 1,152-token chunks, and hot SSD streaming were rejected after end-to-end A/B tests failed to improve the relevant workload.

The packed direct Q6_K path replaces the previous full-matrix FP16 expansion, removes about 118 MB (112.5 MiB) of temporary Metal workspace, and was bit-identical to the prior direct decoder in the isolated A/B. The Q6_K projection improved by about 33% at batches 256–1024; deferring redundant trailing tile barriers improved the fused gate/up projection by about 2.5% with identical FP16 output.

The legacy plain-GGUF gate/up scale buffer uses about 708 MB for this 48-layer model and is built once during engine startup. It remains available for the original layout and the serial diagnostic path, but the default direct-Metal path neither builds nor retains it. A minimal MTP run consequently dropped from about 1.80 GB to about 1.09 GB peak footprint before the larger configured-context allocation is considered.

Attention score scratch is allocated lazily from the real `batch × heads × span`, not from the configured worst-case context. The direct online-attention path, including the default 1,024-token chunk, needs no score scratch through the runtime's 2,048-token batch limit. The buffer grows only for explicit legacy attention or long decode fallbacks.

Sliding-window K/V storage is chunk-safe: its ring capacity is `window + max_batch`, and prefill spans start at the oldest key needed by the first query in the chunk. This matters after the first 1,024 tokens; using only a `window`-sized ring overwrote still-valid keys when a whole chunk was stored before attention and produced an artificially high 2,205-token rate. The corrected path produced the same one-token output hash with 768- and 384-token chunks.

Set `NEUTRON_NO_SSD_PREFETCH=1` to disable the inference engine's cold-start mmap hint or `NEUTRON_NO_RESIDENCY=1` to disable Metal residency sets while diagnosing memory pressure. Persistent packed FFN disables whole-GGUF residency automatically. Metadata inspection and isolated kernel benchmarks deliberately skip whole-file prefetch: several concurrent tools must not each ask macOS to page in the complete 6.86 GiB model. For the tested model on an 18 GB Mac, repeated SSD reads inside the layer/token loop are intentionally avoided.

## Speculative MTP status

Pass a Q4_K_M `gemma4_assistant` GGUF with `--draft-model` to enable greedy MTP. The native graph consumes the target's normalized hidden state, target token embedding and final SWA/global K/V states, then runs the four-layer assistant and verifies one to three proposals in one target pass. The target emits the correction token after a rejection and the bonus token after a fully accepted draft. Target and assistant vocabularies are compared token-for-token before execution.

The tested 12B assistant accepted complete three-token drafts for the repetitive decode probe and produced the same greedy text hash as target-only generation. The adaptive verifier now uses dedicated two-to-four-row W4A16 Q4_K/Q6_K kernels instead of padding the work to a 32-token prefill tile. Its fused Q4_K gate + up + GEGLU kernel writes the intermediate directly as FP16, and the attention/output/down projections use matching FP16-input micro-batch kernels. On the M3 Pro this reduced a representative three-token verification from about 243 ms to **162 ms** and a two-row verification from about 139 ms to **86 ms**. On the final selectively quantized direct-Metal model, the warmed 64-token probe rose from **17.32 token/s target-only** to **21.96 token/s with MTP** (about **26.8%**), with both paths producing hash `4489382881409632950`. The earlier original-layout run measured 16.78 versus 20.61 token/s. `NEUTRON_EXACT_VERIFY=1` disables the faster W4A16 verification path and retains FP32 activations for every verification width. `NEUTRON_SPEC_TRACE=1` reports proposals, accepted counts, correction tokens, and timings.

The assistant checkpoint is already the target-specific small model: about 0.4B parameters, four transformer layers, and 327 MB in Q4_K_M. Gemma 4 E2B is a multi-billion-parameter target model, not a smaller interchangeable drafter for the 12B hidden/KV layout. Replacing the trained 12B assistant with E2B would therefore add work and lose the architecture-specific MTP projection rather than implement the DeepSeek-style auxiliary-module design.

The ordinary prefill path already keeps the heavy weights quantized: Q/K/V/O and gate/up use Q4_K, while mixed-quant down projections and the tied embedding use Q4_K or Q6_K directly. Activations and K/V remain FP16 (W4A16), matching [Google's available 12B QAT deployment format](https://ai.google.dev/gemma/docs/core). A 224-token stage profile attributes about 626 ms to fused gate/up and 321 ms to down projection out of roughly 1.26 s of GPU work. An additional A8 activation path was not enabled: the official 12B format supplies W4A16 rather than calibrated A8 tensors, and Apple M3 `simdgroup_matrix` throughput here is optimized for FP16 tiles; dynamic A8 would add quantization passes without a native matrix path demonstrated to recover that cost.

## Final FFN selection

A 995-token stage profile attributed about 72% of prefill time to gate, up,
and down projections. Q2_K was tested on the dominant gate projection, but
measured 30.84 ms versus 30.68 ms for Q4_K and added substantial numerical
error. It was removed.

Sampling one FFN token in eight across the 40 local layers exceeded 300
token/s (321.8 token/s median), but failed every position in the retrieval
probe and produced unusable output. A stride-four version also scored 0/5.
Neither path is shipped.

The retained compromise performs every FFN operation, uses FP16 accumulation
only in the 40 sliding-window layers, and keeps the eight global layers in
FP32. Together with default stride-two sparse SWA attention it measured about
177.3 token/s on a warm 995-token prompt and scored 5/5 on the retrieval probe.
`--exact-ffn` and `--dense-swa` restore the fully conservative path. The
300-token/s uncached target is therefore not claimed on the tested 18 GB M3
Pro; reaching it requires newer cooperative-tensor hardware or a different
model architecture.

## Current boundaries

- Gemma 4 image/audio/video input is implemented through the matching `mmproj`;
  video additionally requires `ffmpeg` and remote HTTP media must be converted
  to data URLs by the client.
- One graph executes at a time; unrelated-request continuous batching is not implemented.
- One KV sequence is active in Metal memory at a time. Indexed disk snapshots
  can swap reusable prefixes between requests and restarts, but they are not a
  concurrent paged multi-session scheduler.
- The runtime accepts the tested Gemma 4 12B Q4_K_M tensor layout, not arbitrary architectures or quantizations.
- CUDA is an architectural extension point, not a functioning backend in this revision.

Next performance work is focused on continuous batching, paged multi-session KV, GPU-chained draft steps, and model-format changes that reduce memory traffic beyond what direct Q4_K/Q6_K kernels can provide.

## License

The project is MIT licensed. Gemma weights remain subject to Google's model
terms. Ollama is used only as a local model store. Text inference stays in the
native Neutron engine; multimodal builds use a pinned llama.cpp `mtmd`
dependency for image/audio/video projection and llama.cpp also remains the
attention benchmark/design reference.
