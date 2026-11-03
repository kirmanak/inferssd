<!-- SPDX-License-Identifier: Apache-2.0 -->
# Qwen3.8-Flash-Next on an RX 9070 XT

[Back to documentation](README.md)

This note covers the local changes on top of commit `6912c4f`
(`engine, framework: keep a VRAM reserve against this process's own memory`)
and how to run them. Upstream InferSSD documents DeepSeek and Muse on CUDA
and Metal. These changes make the bundled `qwen4exp` graph run on Vulkan,
on one desktop:

| Part | What is installed |
| --- | --- |
| OS | Bazzite, kernel 7.2 |
| CPU | Ryzen 5 7600X, 6 cores |
| RAM | about 64 GB. Runs use a 52 GB cgroup and no swap |
| GPU | AMD Radeon RX 9070 XT, 16304 MiB, RADV GFX1201, PCIe 5.0 x16 |
| SSD | Samsung 980 PRO |
| Checkpoint | Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL`, four GGUF shards, about 111 GB |

The checkpoint is architecture `qwen4exp`: 48 layers, hidden size 2560,
512 routed experts, 10 used per token, expert feed-forward length 640.
Gate and up are Q4_K except layer 2 (Q5_K). Down is Q5_1 except layers
4, 30, 46, and 47 (Q8_0). A token touches 480 experts, about 1.44 GB of
expert weights. The dense model buffer on the GPU is 4606.39 MiB.

Weights used here:

```text
/var/home/kirmanak/models/inferssd/Qwen3.8-Flash-Next-UD-Q4_K_XL/
```

The first shard is `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`.
The expert index is `expert-index/experts.tsv` beside it. Rebuild that
index with `inferssd source index` if you move the shards.

## What changed

The dense layers stay on Vulkan. Routed experts stay in the GGUF until
the host cache admits them. Nothing here copies the full expert set into
VRAM or into a second RAM file.

### The model can finish a Vulkan prompt

Four bugs stopped a normal `llama-completion` run before any expert
speed work mattered.

- Qwen sparse-attention builds a mask by repeating one seed. Vulkan's
  repeat shader is f32 only. Casting the seed to f16 first placed the
  mask on the CPU while the in-place write stayed on the GPU, and the
  run died in `ggml_vk_tensor_subbuffer`. The graph now repeats in f32
  and casts afterwards (`engine/src/models/qwen4exp.cpp`).
- Expert slot bytes were learned from the first routed layer. Later
  layers are wider (Q8_0 down projections). The host cache is sized from
  the widest up, gate, and down role (`engine/src/llama-context.cpp`).
- After load, the cache rebuilt itself from the memory left under the
  52 GB cap. The arena it was about to release was still counted as
  used, so the replacement cache was about 1 GB and the generation
  working set did not fit. That arena is added back into the surplus
  before the resize (`engine/ggml/src/ggml-ssd/host/expert-cache.h`).
  The log line is `resize_keeps_arena=1`.
- `llama-completion` never calls `llama_ssd_prefill_phase`. After a
  multi-token prompt, the next one-token batch is generation and has to
  enter the decode cache phase on its own, before that batch is
  allocated. A phase change also synchronizes. Closing the perf timer
  on that sync recorded an empty window and hid the real compute in
  "unaccounted time". The timer now closes only after `graph_compute`
  has been submitted (`engine/src/llama-context.cpp`).

### Prompt experts that fit run on Vulkan

`ggml_backend_vk_expert_api_v1` exposes the existing quantized Vulkan
matmul for one expert. The CPU prefill asks for the experts that fit in
the device budget, in first-touch order, and computes the rest itself.
Supported types are the ones this GGUF uses: Q4_K, Q5_K, Q5_1, and
Q8_0. Hidden size 640 is not a multiple of 256, which is why this does
not go through the CUDA expert kernels.

An explicit `--ssd-vram-budget` is clamped to the memory still free
after the dense model. `auto` uses that free room minus 128 MiB. The
measured runs pass `268435456` (256 MiB) so a prompt layer still
overflows and the CPU remainder stays non-zero. Staging is one reused
buffer, a few megabytes, not one allocation per layer.

One-token decode does not use this path. A failed matmul is computed
again on the CPU. The unit test
`engine/tests/ssd/test-vulkan-expert.cpp` drives the real prefill entry
with hidden size 640, checks expert 0 against the CPU math, and checks
that expert 1 stays on the CPU sink.

### Generation uses the host cache in one batch per layer

When every expert selected for a one-token layer is already in the host
cache, the sink runs one up/gate pass and one down pass for the whole
layer. A miss stays on the per-expert reader. The log line is
`[kimi_cpu_sink] resident batch`.

Over a long reply the working set of this checkpoint is about 32 GB
across 256 tokens, and the host cache settles near 50 GB, so almost all
of those reads hit RAM. A token still moves about 1.44 GB. The CPU
walks that in about 38 ms. The dense GPU shaders for one token are
about 33 ms. Those two are sequential across the 48 layers, which is
why a warm token is near 90 ms.

## Measured rates

Same machine, same 507-token prompt, `GGML_VK_VISIBLE_DEVICES=0`,
`MemoryMax=52G`, `MemorySwapMax=0`, `-ngl 99`, `-sm none`,
`--device Vulkan0`, `-c 2048`, `-fa on`, `--mmap --no-repack --fit off
--no-warmup --defer-experts`, no `--cpu-moe`.

| Run | Prompt | Generation |
| --- | --- | --- |
| 1024 new tokens, resident batch | 58.77 tok/s (8627.49 ms / 507) | 10.68 tok/s (95816.43 ms / 1023 runs, 93.66 ms) |
| 128 new tokens, run 1, current tree | 56.34 tok/s | 8.61 tok/s (127 runs) |
| 128 new tokens, run 2, current tree | 58.29 tok/s | 8.76 tok/s (127 runs) |

The short runs are slower because the first tokens still miss the host
cache. On the long run the sink's own wall time falls from about 68 ms
on the first 127 tokens to about 48 ms on the last 127. Both short runs
exited 0, kept 49/49 layers on Vulkan0, kept the model buffer at
4606.39 MiB, and were not oom-killed. Prompt logs show 4030 GPU
experts with `gpu_expert_us` greater than zero on 47 of 48 layers, and
8764 experts still computed by the CPU sink. One-token cache hits were
51737 and 52549. Expert staging peaked at 6.51 MiB and assigned weights
at 255 MiB.

The parity test, run twice under `MemoryMax=4G`, printed the same line
both times:

```text
VULKAN_EXPERT_MATCH hidden=640 up=q4_K down=q8_0 rel_l2=0.00347405 gpu_ids=0 cpu_ids=1 device_bytes=188416 budget=358400
```

A gather of every selected expert into one Vulkan submission was timed
and not implemented. Device-local shaders are about 33 ms, and reading
1.44 GB from host memory the GPU can import is about 41 ms at the
measured 35 GB/s, so that design lands near 74 ms (about 13.5 tok/s)
only when every expert is already in anonymous RAM. File-backed GGUF
mappings cannot be imported (`VK_ERROR_INVALID_EXTERNAL_HANDLE`). About
9 in 10 warm tokens still miss the host cache once, so filling a miss
with zeros would be the wrong token. A 10 GB VRAM set of hot experts
would have covered about 72% of the slots in the last 256 tokens of the
long run. That path is not in this tree, and 20 tok/s has not been
measured.

## Build

The host install has `libvulkan` and no Vulkan headers. Compile inside
the `inferssd` distrobox. Run the binary on the host. `systemd-run` is
not available inside the container.

```sh
distrobox enter inferssd -- bash -lc '
  cd /path/to/this/checkout &&
  cmake --preset linux-cpu -DGGML_VULKAN=ON -B build-linux-cpu &&
  cmake --build build-linux-cpu -j6 --target llama-completion inferssd-server test-vulkan-expert
'
```

`linux-cpu` leaves CUDA off and turns the SSD store on. `-DGGML_VULKAN=ON`
is required. The preset alone does not enable it. The binaries land in
`build-linux-cpu/bin/`.

The Vulkan parity test needs the 9070, not the Raphael iGPU:

```sh
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \
  env GGML_VK_VISIBLE_DEVICES=0 \
  ./build-linux-cpu/bin/test-vulkan-expert
```

`test-completion-cache-phase` is built with the tests and is not a
CTest entry. It wants a real shard, the expert TSV, and a log path:

```text
test-completion-cache-phase MODEL EXPERTS_TSV LOG
```

## One completion

`-c` has to hold the prompt plus `-n`. The file below is 507 tokens, so
`-c 2048` covers `-n 1024`. Raise `-c` for a longer prompt. Do not
pre-create `--ssd-telemetry`. The server and the completion tool create
that file themselves and refuse an existing one.

```sh
systemd-run --user --scope -p MemoryMax=52G -p MemorySwapMax=0 -- \
  env GGML_VK_VISIBLE_DEVICES=0 \
  ./build-linux-cpu/bin/llama-completion \
  -m /var/home/kirmanak/models/inferssd/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --ssd-expert-source '["/var/home/kirmanak/models/inferssd/Qwen3.8-Flash-Next-UD-Q4_K_XL/expert-index/experts.tsv"]' \
  --ssd-vram-budget 268435456 \
  --mmap --no-repack --fit off --no-warmup --defer-experts \
  -ngl 99 -sm none --device Vulkan0 \
  -c 2048 -b 512 -ub 512 -t 6 -tb 6 -np 1 -fa on \
  -n 128 -f /path/to/prompt.txt --log-colors off
```

`GGML_VK_VISIBLE_DEVICES=0` hides the integrated GPU. Without the 52 GB
scope the host budget is refused and the process exits before prefill.
`--cpu-moe` bypasses this path. `-t 12` is not what these runs used.

## OpenCode

`llama-completion` prints one reply and exits. OpenCode needs the
OpenAI-compatible server, which this build calls `inferssd-server`.
Start it on the host and leave it running. A coding session wants a
larger context than the bench. 32768 fits the card for this model
because most layers are recurrent. Drop it if the server cannot
allocate.

```sh
systemd-run --user --scope -p MemoryMax=52G -p MemorySwapMax=0 -- \
  env GGML_VK_VISIBLE_DEVICES=0 \
  ./build-linux-cpu/bin/inferssd-server \
  -m /var/home/kirmanak/models/inferssd/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --ssd-expert-source '["/var/home/kirmanak/models/inferssd/Qwen3.8-Flash-Next-UD-Q4_K_XL/expert-index/experts.tsv"]' \
  --ssd-vram-budget 268435456 \
  --mmap --no-repack --fit off --no-warmup --defer-experts \
  -ngl 99 -sm none --device Vulkan0 \
  -c 32768 -b 512 -ub 512 -t 6 -tb 6 -np 1 -fa on \
  --alias qwen3.8-flash \
  --host 127.0.0.1 --port 8080 \
  --api-key local \
  --reasoning off --jinja
```

Check the id the server publishes:

```sh
curl -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer local'
```

OpenCode 2.2.5 on this machine reads `~/.config/opencode/opencode.json`.
Add a provider next to the existing ones, and add `inferssd` to
`enabled_providers`. The model key must match `--alias`.

```json
"inferssd": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "InferSSD (local)",
  "options": {
    "baseURL": "http://127.0.0.1:8080/v1",
    "apiKey": "local"
  },
  "models": {
    "qwen3.8-flash": {
      "name": "Qwen3.8 Flash Next",
      "limit": { "context": 32768, "output": 4096 }
    }
  }
}
```

Restart OpenCode. `/models` lists it under InferSSD. The default model
stays whatever `model` already says until you set
`"model": "inferssd/qwen3.8-flash"`.

Only one server can run. It holds the GPU and about 50 GB of RAM.
Generation is about 10 tokens per second once the cache is warm. A
coding turn sends the whole conversation again, and prefill on this
checkpoint is about 55 tokens per second, so a long tool loop spends
minutes before the first new token.
