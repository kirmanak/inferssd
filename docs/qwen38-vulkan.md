<!-- SPDX-License-Identifier: Apache-2.0 -->
# Qwen3.8-Flash-Next on an RX 9070 XT

[Back to documentation](README.md)

This note covers the local changes on top of commit `6912c4f`
(`engine, framework: keep a VRAM reserve against this process's own memory`)
and how to run them. That commit's documentation covers DeepSeek and Muse
on CUDA and Metal. These changes make the bundled `qwen4exp` graph run on
Vulkan, on one desktop:

| Part | What is installed |
| --- | --- |
| OS | Bazzite, kernel 7.2.9 |
| CPU | Ryzen 5 7600X, 6 cores / 12 threads |
| RAM | about 62 GiB. Runs use a 52 GB cgroup and no swap |
| GPU | AMD Radeon RX 9070 XT, 16304 MiB, RADV GFX1201, PCIe 5.0 x16 |
| SSD | Samsung 980 PRO |
| Checkpoint | Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL`, four GGUF shards, 111,334,654,784 bytes (103.68 GiB, 5.03 BPW) |

The GGUF `general.size_label` is `512x56B`. The loader reports architecture
`qwen4exp`, model type A3B, 176.94 B parameters, file type Q4_K Medium.
Tensor census from a load of this file: f32 557, q8_0 503, q4_K 94, q5_1 43,
bf16 24, q5_K 2, iq4_nl 1.

Geometry: 48 layers, hidden size 2560, 512 routed experts, 10 used per
token, expert feed-forward length 640, training context 262144. Gate and up
are Q4_K (921600 bytes) except layer 2, which is Q5_K (1126400 bytes). Down
is Q5_1 (1228800 bytes) except layers 2, 4, 30, 46, and 47, which are Q8_0
(1740800 bytes). The expert index has 73728 rows, 48 layers times 512
experts times the three roles. A token touches 480 experts, about 1.44 GB.
Shared experts (`ffn_*_shexp`) stay with the dense GPU weights. The
per-layer token embedding `per_layer_token_embd.weight` is IQ4_NL and stays
mmap'd. The dense model buffer on the GPU is 4606.39 MiB. Deferred experts
are about 71.73 GiB. The dense stack reported at load is about 31.95 GiB,
which includes mmap'd CPU-side weights, so it is larger than the 4606.39 MiB
Vulkan buffer.

Weights used here:

```text
/var/home/kirmanak/models/inferssd/Qwen3.8-Flash-Next-UD-Q4_K_XL/
```

The first shard is `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
(10946624 bytes, metadata and tokenizer, 0 tensors). The other three shards
hold the weights. The expert index is `expert-index/experts.tsv` beside the
shards. Rebuild that index with `inferssd source index` if you move them.
The measured prompt is
`/var/home/kirmanak/.grok/long-running-background-tasks/inferssd-prompt.txt`
(507 tokens).

## What changed

The dense layers stay on Vulkan. Routed experts stay in the GGUF until the
host cache admits them. Nothing here copies the full expert set into VRAM
or into a second RAM file.

### The Vulkan prompt can finish

The mask bug was the one that aborted a run. The other three changes fix
the timer and the host-cache size. All four are in this tree.

- Qwen sparse-attention builds a mask by repeating one seed. Vulkan's
  repeat shader is f32 only. Casting the seed to f16 first placed the mask
  on the CPU while the in-place `set_rows` stayed on the GPU. The run
  SIGSEGV'd in `ggml_vk_tensor_subbuffer` after `ssd.prefill.layer` had
  succeeded for layers 0, 1, and 2. Layer 2 had read 1169719296 bytes, and
  stdout had already printed the blk.1 and blk.2 source reads. There was
  no layer 3 event. Layers 0–2 KV `set_rows` were fused and did not hit
  that path. The graph now repeats in f32 and casts afterwards when
  `type_mask` is not f32 (`engine/src/models/qwen4exp.cpp`). The seed stays
  a one-element view of the layer score so the fill remains on that
  device. `-fa off` was already the f32 path.
- Expert slot bytes were learned from the first routed layer. Later layers
  are wider because of the Q8_0 down projections. Learning the real slot
  on the first decode token released the populated arena and mapped a new
  one during generation. The host cache is sized from the widest up, gate,
  and down role (`engine/src/llama-context.cpp`).
- After load, the cache rebuilt itself from the memory left under the
  52 GB cap. The arena it was about to release was still counted as used,
  so the replacement cache was about 1 GB and the generation working set
  did not fit. That arena is added back into the surplus before the resize
  (`engine/ggml/src/ggml-ssd/host/expert-cache.h`). The log line is
  `resize_keeps_arena=1`.
- `llama-completion` never calls `llama_ssd_prefill_phase`. After a
  multi-token prompt, the next one-token batch is generation and has to
  enter the decode cache phase on its own, before that batch is allocated.
  A phase change also synchronizes. Closing the perf timer on that sync
  recorded an empty window: the first smoke printed millions of tok/s
  because prompt eval was 0.00 ms. The timer now closes only after
  `graph_compute` has been submitted (`engine/src/llama-context.cpp`). A
  short smoke after that fix printed 9.13 and 3.26 tok/s. That smoke is
  not one of the rows below.

### Prompt experts that fit run on Vulkan

`ggml_backend_vk_expert_api_v1` exposes the existing quantized Vulkan
matmul for one expert. The CPU prefill asks for the experts that fit in
the device budget, in first-touch order, and computes the rest itself.
Supported types are the ones this GGUF uses: Q4_K, Q5_K, Q5_1, and Q8_0.
Each selected expert is three `ggml_mul_mat` calls, each synchronized.
SwiGLU stays on the CPU.

An explicit `--ssd-vram-budget` is clamped to the memory still free after
the dense model. `auto` uses that free room minus 128 MiB. Budget `0`
leaves every expert on the CPU sink and is the baseline in the table.
The comparison runs pass `268435456` (256 MiB) so a prompt layer still
overflows and the CPU remainder stays non-zero. On this file that budget
holds about 87 experts in a layer. Staging is one reused buffer. The cap
is 64 MiB. The current-tree runs peaked at 6.51 MiB staged and about
255 MiB assigned.

One-token decode does not use this path. `select_vulkan` returns when the
batch has one token. A failed matmul is computed again on the CPU. The
unit test `engine/tests/ssd/test-vulkan-expert.cpp` drives the real
prefill entry with hidden size 640, checks expert 0 against the CPU math,
and checks that expert 1 stays on the CPU sink.

On the 256 MiB runs the prompt still read every expert from the SSD, about
40.2e9 bytes and about 7 s at about 5.8 GB/s. The prompt rate did not rise.
The first of those two runs also paid for pipeline compiles.

### Generation uses the host cache in one batch per layer

When every expert selected for a one-token layer is already in the host
cache, the sink runs one up/gate pass and one down pass for the whole
layer. A miss stays on the per-expert reader. The log line is
`[kimi_cpu_sink] resident batch`.

Over the last 256 tokens of the long run the working set is about 10310
unique experts, about 32 GB. The host cache settles near 50 GB under the
52 GB cap. About 1% of the slots on a warm token still miss. Those misses
are spread across tokens: 26 of the last 256 tokens hit every slot, and 2
of the last 64 did. The first miss usually lands around layer 8 to 11.
Filling a miss with zeros would be the wrong token.

A token still moves about 1.44 GB. The CPU walks that in about 38 ms. The
dense GPU shaders for one token are about 33 ms (33.40 ms of decode-shaped
GPU time on the timestamp logger). The long run averages 93.66 ms per
generated token, so about 10 ms more sits in the handoff. The 48 layers
run one after another. The sink's own wall time on that run falls from
68.26 ms on the first 127 tokens to 47.67 ms on the last 127.

## Measured rates

Same machine, same 507-token prompt, `GGML_VK_VISIBLE_DEVICES=0`,
`MemoryMax=52G`, `MemorySwapMax=0`, `-ngl 99`, `-sm none`,
`--device Vulkan0`, `-c 2048`, `-fa on`, `--mmap --no-repack --fit off
--no-warmup --defer-experts`, `-t 6`, no `--cpu-moe`. `llama-completion`
counts the first generated token inside prompt eval, so `-n 128` prints
127 generation runs and `-n 1024` prints 1023.

| Run | Prompt | Generation |
| --- | --- | --- |
| CPU sink, budget 0, mask and timer only | 54.02 tok/s (9386.13 ms / 507) | 3.31 tok/s (38322.92 ms / 127) |
| same, second run | 54.24 tok/s (9347.32 ms / 507) | 3.32 tok/s (38294.71 ms / 127) |
| host-cache fixes, budget 0 | 52.91 tok/s (9582.20 ms / 507) | 8.75 tok/s (14513.00 ms / 127) |
| same, second run | 53.16 tok/s (9537.92 ms / 507) | 8.78 tok/s (14472.73 ms / 127) |
| Vulkan prompt experts, 256 MiB, cold pipelines | 47.79 tok/s (10609.18 ms / 507) | 8.79 tok/s (14452.50 ms / 127) |
| same, second run | 54.24 tok/s (9346.83 ms / 507) | 8.85 tok/s (14352.83 ms / 127) |
| current tree, 128 new tokens | 56.34 tok/s (8999.24 ms / 507) | 8.61 tok/s (14751.48 ms / 127) |
| same, second run | 58.29 tok/s (8697.35 ms / 507) | 8.76 tok/s (14496.32 ms / 127) |
| resident batch, 1024 new tokens | 58.77 tok/s (8627.49 ms / 507) | 10.68 tok/s (95816.43 ms / 1023 runs, 93.66 ms) |

The short runs are slower because the first tokens still miss the host
cache. 10.68 tok/s is the best measured generation rate. The two
timestamp-logger prompt passes printed 53.99 and 54.80 tok/s. Their decode
lines are a logger pass of one cold token, not a generation rate.

The budget-0 pair exited 0 with 49/49 layers on Vulkan0, the model buffer
at 4606.39 MiB, sink reads of 233450582016 bytes, `read_errors=0`, and
about 0.6% unaccounted time. Host-cache decode hits on that next pair were
51924 and 51953. The 256 MiB pair's one-token hits were 52330 and 52617,
with 4030 GPU experts and 8764 still on the CPU sink. The current-tree
remeasure (same source as the 17:22 binary, run later the same day) exited
0 on both tries, again 49/49 on Vulkan0 and 4606.39 MiB, with GPU experts
summed to 4030 and `gpu_expert_us` greater than zero on 47 of 48 layers.
Layer 47 staged no GPU experts. One-token hits on that pair were 51737 and
52549. Those hit counts belong to that pair.

The parity test, run twice under `MemoryMax=4G`, printed the same line
both times:

```text
VULKAN_EXPERT_MATCH hidden=640 up=q4_K down=q8_0 rel_l2=0.00347405 gpu_ids=0 cpu_ids=1 device_bytes=188416 budget=358400
```

A finished run can print this and still exit 0. The long run's copy was
0.8047 MiB. A later logger run printed 0.8008 MiB. `sched_reserve` prints
0.11 MiB. The mismatch is not diagnosed.

```text
~llama_context: Vulkan_Host compute buffer size of 0.8008 MiB, does not match expectation of 0.1094 MiB
```

A gather of every selected expert into one Vulkan submission was timed
and not implemented. Device-local shaders are about 33 ms, and reading
1.44 GB from host memory the GPU can import is about 41 ms at the
measured 35 GB/s, so those two add up near 74 ms (about 13.5 tok/s) only
when every expert is already in anonymous RAM. File-backed GGUF mappings
cannot be imported (`VK_ERROR_INVALID_EXTERNAL_HANDLE`,
`vkAllocateMemory` -1000072003). An empty microbench of 48 separate
submits was about 82 ms. A real token does not pay that 82 ms. The
expert matmul itself is about 9 ms and sits under the transfer. A 10 GB
VRAM set of hot experts, replayed and not built, would have covered about
72% of the slots in the last 256 tokens and none of those tokens
completely. 20 tok/s has not been measured.

## Build

The host install has `libvulkan` and no Vulkan headers. Mesa on the
measured machine was 26.2.4 with vulkan-loader 1.4.341. Compile inside
the `inferssd` distrobox, which has the headers, g++, and `glslc`. Run
the binary on the host. `systemd-run` is not available inside the
container, and a run without the scope is refused (`host budget refused`).

```sh
distrobox enter inferssd -- bash -lc '
  cd /path/to/this/checkout &&
  cmake --preset linux-cpu -DGGML_VULKAN=ON -B build-linux-cpu &&
  cmake --build build-linux-cpu -j6 --target llama-completion inferssd-server test-vulkan-expert
'
```

`linux-cpu` leaves CUDA off, turns native CPU flags on, and turns the SSD
store on. `-DGGML_VULKAN=ON` is required. The preset alone does not enable
it. The binaries land in `build-linux-cpu/bin/`. The checkout used for the
rates above does not have to contain `build-linux-cpu`. The measured
binary was built from this source at 17:22 on 2026-11-03.

The Vulkan parity test needs the 9070, not the Raphael iGPU. Raphael
reports system RAM as a larger device-local heap, so a picker that wants
the biggest heap selects it unless the variable below hides it.

```sh
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \
  env GGML_VK_VISIBLE_DEVICES=0 \
  ./build-linux-cpu/bin/test-vulkan-expert
```

`test-completion-cache-phase` is built with the tests and is not a CTest
entry. It wants a real shard, the expert TSV, and a log path. It drives
`llama_decode` through `llama_batch_get_one` and does not call
`llama_ssd_prefill_phase`.

```text
test-completion-cache-phase MODEL EXPERTS_TSV LOG
```

The Vulkan test wants the 9070 and a small btrfs fixture. On this machine
that fixture is `/var/home/kirmanak/models/inferssd/vulkan-expert-test`.

## One completion

`-c` has to hold the prompt plus `-n`. The measured prompt is 507 tokens,
so `-c 2048` covers `-n 1024` (1531 tokens of context). Training context
is 262144. Leaving `-c` at 0 asks for that and tries to allocate it. At
`-c 2048` the log shows a 48.00 MiB KV cache for 12 full-attention layers,
a 12.00 MiB indexer KV cache, and 112.57 MiB of recurrent state across all
48 layers. Do not pre-create `--ssd-telemetry`. The server and the
completion tool create that file themselves and refuse an existing one.

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
  -n 128 \
  -f /var/home/kirmanak/.grok/long-running-background-tasks/inferssd-prompt.txt \
  --log-colors off
```

`GGML_VK_VISIBLE_DEVICES=0` hides the integrated GPU. Without the 52 GB
scope the host budget is refused and the process exits before prefill.
`--cpu-moe` bypasses this path. `-t 12` is not what these runs used.
`INFERSSD_DEBUG=profile` was off for the rate runs.

## OpenCode

`llama-completion` prints one reply and exits. OpenCode needs the
OpenAI-compatible server, which this build calls `inferssd-server`.
Start it on the host and leave it running. The bench context 2048 is
small for a coding session. The command below asks for 32768. That length
was not timed. The measured cache sizes above are for 2048. Drop 32768 if
the server cannot allocate. The server's own default timeout is 600 s.

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

OpenCode 2.2.5 on this machine is
`/home/linuxbrew/.linuxbrew/bin/opencode` and reads
`~/.config/opencode/opencode.json`. The file on this machine uses a
singular `provider` object, `enabled_providers`, and a default model.
Add a provider next to the existing ones, and add `inferssd` to
`enabled_providers`. The model key must match `--alias`. This snippet is
not written into that file by the checkout.

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
Generation on the long bench is 10.68 tok/s once the cache is warm.
A coding turn sends the whole conversation again. Prompt eval on this
checkpoint is about 53 to 58 tok/s, so a long tool loop spends its time
before the first new token.
