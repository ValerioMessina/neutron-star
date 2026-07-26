#!/usr/bin/env python3
"""Small HTTP benchmark reporting TTFT and streamed token-piece throughput."""
import argparse, json, os, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
p.add_argument("--prompt", default="Explain why prefix caching reduces LLM latency.")
p.add_argument("--tokens", type=int, default=128)
a = p.parse_args()
body = json.dumps({"model":"gemma-4-12b", "messages":[{"role":"user","content":a.prompt}],
                   "max_tokens":a.tokens, "temperature":0, "stream":True}).encode()
headers = {"Content-Type":"application/json"}
if os.getenv("NEUTRON_API_KEY"): headers["Authorization"] = "Bearer " + os.environ["NEUTRON_API_KEY"]
start = time.perf_counter(); first = None; pieces = 0; usage = {}
with urllib.request.urlopen(urllib.request.Request(a.url, body, headers)) as r:
    for raw in r:
        if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]": continue
        event = json.loads(raw[6:]); usage = event.get("usage") or usage
        delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            first = first or time.perf_counter(); pieces += 1
end = time.perf_counter()
generated = usage.get("completion_tokens", pieces)
decode_tokens = max(generated - (1 if first else 0), 0)
print(json.dumps({"ttft_ms": round(1000*((first or end)-start), 2), "seconds": round(end-start, 3),
                  "prompt_tokens": usage.get("prompt_tokens"), "generated_tokens": generated,
                  "stream_pieces": pieces, "decode_tokens_per_second": round(decode_tokens/max(end-(first or end), 1e-9), 2)}))
