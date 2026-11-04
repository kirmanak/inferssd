<!-- SPDX-License-Identifier: Apache-2.0 -->
# Prefill toward 100 tok/s

Journal for the Qwen3.8-Flash-Next Vulkan prefill work on the RX 9070 XT.
The measured prompt is 507 tokens. Baseline is the resident run on 2026-11-03:
prompt eval 8627.49 ms, 58.77 tok/s. Decode on that run was 10.68 tok/s and
must not get slower. Quality means the same greedy tokens as the CPU sink,
or a recorded numeric difference no larger than the existing Vulkan matmul.

## Where the time is

From `inferssd-resident-1024.log`, 48 prefill layers:

- Expert reads: 40.155 GB in 6.985 s, 5.75 GB/s. O_DIRECT, so the page cache
  does not keep them.
- Prompt wall: 8.627 s. The read is already most of that. Hiding the other
  1.6 s (dense graph and gaps) tops out near 73 tok/s.
- A Samsung 980 PRO cannot move 40.155 GB in the 5.07 s that 100 tok/s
  requires (that would be 7.9 GB/s). A cold prompt cannot hit 100 tok/s
  on this drive without reading fewer bytes.
- Layer 0, the only `kimi_cpu_sink_layer` line: compute 190 ms, IO 219 ms,
  wait 31 ms, wall 237 ms. Once the bytes are in RAM the CPU GEMM is still
  about as long as the read. A warm CPU prefill stays near 70 tok/s.
- The per-expert Vulkan path does not raise the prompt rate. Each expert is
  three synchronized matmuls, and the prompt still reads every expert from
  the SSD. Summed `gpu_expert_us` was 2.7 s and changed nothing.

The host cache is the way through the drive ceiling. At load this machine
mapped 13316 slots, 49.7 GiB, `slot_bytes=4005888`, and left them empty
during prefill (`valid_bytes=0`, `prefill_misses=12794`). Production already
has `cache_prefill_read`. A second prompt in the same process can hit those
slots if the first prompt admits into them. Uniform allocation gives about
277 slots per layer. Layer 0 selected 344 experts, so some layers still miss.
Those misses are a few hundred megabytes, not 40 GB.

100 tok/s then needs the warm compute to be a batched GPU matmul, not the
CPU sink. A layer is about 0.8 GB. At the measured 35 GB/s host import that
is ~25 ms of traffic, plus the matmul. 48 layers at 60 ms plus the ~1.6 s
dense graph is about 4.5 s, 110 tok/s, if the batch submission is not itself
the bottleneck. File-backed GGUF cannot be imported. The cache arena is
anonymous RAM, so the copy is an ordinary staging upload.

## Tried

1. Admit prefill misses into resident cache slots, read straight into the
   slot, publish after the consumer. A second prompt hits. Cold still
   streams. Waiting for every miss deadlocks the 8-slot ring, so misses
   keep streaming and only an all-hit layer takes the batch.
2. One Vulkan graph for the up/gate pairs and one for the down projections.
   `INFERSSD_PREFILL_GPU=0` disables it. Outputs come back on one fence.
   Per-tensor fences, a 1 GB packed upload, mapped BAR reads, and a
   pinned-host graph were slower or crashed. 512-expert chunks did not
   beat 96 on the server.
3. Import the cache arena with `VK_EXT_external_memory_host` so the GEMM
   reads weights in place. Not wired in. See the bench below.

Not the path to 100:

- Sorting cold reads can approach 7 GB/s, about 70 tok/s cold.
- A fused SwiGLU removes a CPU round trip inside the expert batch. That is
  part of the 0.6 s of batch overhead, not the 1.7 s still missing.

## Results

Second prompt in one process, 500 tokens from the 507-token file, commit
`ec9a46d`. Decode was not remeasured: `n_predict` 1 does not time
generation, and this path is not on the decode graph.

| | time | tok/s |
| --- | --- | --- |
| cold | 8516 ms | 58.7 |
| warm | 6707 ms | 74.55 |

Warm split from the batch timers: expert batch 3.69 s (upload 1.77, GEMM
0.87, one-fence download 0.42), and 3.02 s of dense graph plus scheduler.
138 SSD misses are not the limiter. 100 tok/s is 5.00 s, so 1.71 s has to
come off this split.

Host import works. A 4096-byte-aligned anonymous mapping imports through
1 GB (`caps.buffer_from_host_ptr` is still false; the function pointer is
set). Q4_K, K=2560, N=640, 16 tokens, same weights, samples match
(`-0.00152 0.00486`):

| experts | weight | staged | import | ratio |
| --- | --- | --- | --- | --- |
| 64 | 59 MB | 6.18 ms | 4.38 ms | 0.71 |
| 256 | 236 MB | 24.7 ms | 24.3 ms | 0.99 |
| 512 | 472 MB | 51.9 ms | 47.9 ms | 0.92 |

At 512 the upload drops from 17.8 ms to 2.8 ms and the GEMM grows from
28.7 ms to 39.8 ms. The shader reads the imported weights over PCIe, so
the copy is not removed, it moves. Scaled to 40 GB the save is about
0.33 s, roughly 78 tok/s. Not worth the path.

100 tok/s needs the 1.77 s upload to overlap the 3.02 s dense graph, or
the dense graph to shrink by about 1.7 s. The router runs after attention
in the same split, so the selected experts are not known in time to hide
that layer's upload in that layer's attention. Upload the cached experts
of layer N+1 at the end of layer N, on a queue that overlaps attention.

A 1.5 GB copy against a 44 ms compute shader on the RX 9070 XT:

| queues | copy | compute | both | overlap |
| --- | --- | --- | --- | --- |
| two async-compute queues (ggml's fallback pair) | 165 ms, 9.8 GB/s | 32 ms | 169 ms | 14% |
| graphics queue copy, async-compute shader | 55 ms, 29 GB/s | 47 ms | 54 ms | full |

There is no transfer-only family. The graphics queue is the DMA engine,
and it does overlap async compute. ggml already prefers that family for
the transfer queue when the compute queue is the non-graphics one. The
current upload still waits inside the expert callback, after attention
has finished, so the DMA never gets that window. A CPU memcpy of the
layer into a staging buffer before the DMA would put ~15 ms back on the
critical path. The copy has to read the cache arena.

Tried that copy. At the end of layer N, import the next layer's cached
slots and DMA them on the graphics queue into a device buffer. The next
layer's matmul skipped the upload (`set_us` fell to 48). The GEMM then
took 150–400 ms for ~70 experts instead of ~15 ms, and later chunks went
past a second. A concurrently shared buffer, so the copy and the shader
did not need a queue-family acquire, did not change that. The 500-token
request did not finish in 7 minutes. That wiring is not in the tree.

The isolated sequence is fine. 512 Q4_K experts, K=2560 N=640, 16 tokens,
exclusive device buffer, graphics-queue copy, one acquire, then the
existing mul_mat:

| | time |
| --- | --- |
| staged set | 15.0 ms |
| staged GEMM | 6.6 ms |
| one-blob copy | 15.3 ms |
| acquire | 0.16 ms |
| GEMM after acquire | 6.3 ms, then 6.1 ms |
| sample error | 0 |

So a shader can read an exclusive buffer the graphics queue just filled,
once the compute queue has acquired it. What failed in the server was
the wiring (concurrent buffer, no acquire, copies still in flight), not
the GEMM. 512 separate region copies from an imported mapping were not
the same: 124 ms to copy 472 MB, and the first GEMM was 30 ms. One
contiguous copy is the usable shape. It is not faster than the current
per-tensor set when done synchronously (both ~15 ms / 472 MB). It only
pays if the copy runs during the next layer's attention.

A copy whose source is an imported host mapping does not keep that GEMM.
One region, 472 MB, from an imported pointer into the same exclusive
buffer: copy 125 ms, GEMM 28.9 ms and 28.7 ms on the next run, sample
still exact. The staging-buffer copy is the one that leaves the matmul
at 6 ms. Do not DMA expert weights out of an imported mapping.

The staging copy can run while another GEMM is in flight.
`overlap_compute_us=6231` against `staged_compute_us=6433` for the same
512-expert graph, with a 472 MB graphics-queue upload outstanding. The
memcpy into the uncached staging buffer is the part that does not hide:
58–103 ms, slower than the current per-tensor set (15 ms). Replacing
that set with this upload is a loss unless the copy is already in flight
during attention and the memcpy is not on the critical path.

## Dense graph is not the 3 seconds

`GGML_VK_PERF_LOGGER=1` on the warm 500-token prompt. The logger itself
slows the prompt (7564 ms, 66.1 tok/s, versus 6829 ms without it). GPU
timestamps for that pass, 2.20 s total:

| | GPU time |
| --- | --- |
| prompt-sized matmul, n=496 | 0.31 s |
| of which the largest is f32 m=4 n=496 k=10240 | 0.11 s |
| FLASH_ATTN_EXT | 2.6 ms |
| everything that is not MUL_MAT | 0.21 s |
| quantized expert matmuls | 1.33 s |

There is no 1.7 s inside the dense kernels. The 3 s beside the expert
timer is CPU: recording, scheduling, and the upload that still waits
inside the expert callback. Hiding that upload during attention is still
the way to 100 tok/s, and only with a staging source, an exclusive
buffer, and one acquire before the shader.

## Ahead upload of the whole next layer lost

At the end of layer N, pack every cached expert of layer N+1 and
`memcpy` that into a staging buffer, then one graphics-queue copy.
Not the per-tensor set. The packed copy is 60 ms / 472 MB; the set is
15 ms for the same bytes. The pack also copied the whole cached layer,
about 37 GB across the prompt, not the routed subset.

| where the memcpy ran | warm wall | tok/s |
| --- | --- | --- |
| layer thread | 45437 ms | 11.00 |
| background thread | 22601 ms | 22.12 |

Keys matched (`used` 33666), so skipping `tensor_set` worked. On the
background run the DMA hid (`finish_us` sum 0.03 s) and the memcpy sum
was 2.85 s. Resident expert time grew to 5.01 s. The memcpy ran during
the dense graph and the wall grew by more than the memcpy. That wiring
is not in the tree. Reverted, rebuilt, same 500-token prompt:

| | time | tok/s |
| --- | --- | --- |
| cold | 8563 ms | 58.4 |
| warm | 6573 ms | 76.07 |

That is the clean number. `/completion` with `n_predict` 32 stopped at
`eos` (`predicted_n` 1, empty content). `llama-completion -n 128` on
the reverted binary, same prompt and flags as `docs/qwen38-vulkan.md`:

```text
prompt eval time = 8923.50 ms / 507 tokens (56.82 tok/s)
eval time = 12176.46 ms / 127 runs (95.88 ms per token, 10.43 tok/s)
```

The published decode figure is 10.68 tok/s over 1023 runs. This sample
is 127 runs and still cold on many experts. It is not a broken decode
path. The HTTP server is the wrong tool: it sampled `eos` immediately.

Per-tensor staging copies, submitted once and acquired once, do not
replace the set. 512 Q4_K experts, 472 MB, each region memcpy'd into
its own slice of a cached staging buffer (`flags=0xe`), one submit, one
acquire, then the existing matmul:

| | time |
| --- | --- |
| per-tensor set | 15.0 ms |
| slice memcpy | 54 ms |
| acquire | 0.2 ms |
| GEMM after acquire | 6.5 ms, then 6.1 ms, sample error 0 |

The GEMM is fine. The copy is not. A 472 MB staging buffer write-allocates;
the 15 ms set uses a small staging buffer that stays in cache. Submitting
the slow copy without waiting does not make it the fast one. Not wired in.

## Background small-staging set

At the end of layer N, a background thread calls the existing
`ggml_backend_tensor_set` for every cached expert of layer N+1. That is
the 15 ms / 472 MB path, not a packed memcpy. The next layer joins before
it touches the cache, then skips `tensor_set` for a matching pointer.
Decode does not start the thread (`ntokens` must be above 32).

Two warm 500-token prompts on that binary:

| | time | tok/s |
| --- | --- | --- |
| warm 1 | 6183 ms | 80.87 |
| warm 2 | 6298 ms | 79.39 |

Both beat 6573 ms. On the first of those, the background set moved 37.6 GB
in 1.62 s and the batch skipped 34059 uploads. The join still waited
0.92 s, so the upload is only partly hidden. Expert-callback time fell
from about 3.7 s to 2.25 s. No fallback.

`llama-completion -n 128` on the same binary: 9.72 tok/s, then 10.77.
The clean binary's short run was 10.43. The 9.72 sample is noise; 10.77
is above the 10.68 baseline. The thread does not run on that tool's
decode path. Kept.

The join was the copy that did not fit in the gap after the batch. The
next layer's set now starts at the beginning of this layer, into the
other of two buffers, while this layer's batch still reads the first.
Those cache slots are pinned until the copy finishes. The batch must not
join that in-flight fill: doing so waits for the next layer and then
shows its slot to this batch (`used` fell to 30, wall 7337 ms).

After that join was removed:

```text
prompt_ms=5803.14 prompt_per_second=86.16
join sum 0.123 s, max 64 ms
used 34059
batch set 0.42 s, compute 0.85 s
```

`llama-completion -n 128` on that binary: 10.79 tok/s. Above 10.68.
5803 ms is not 5000. The remaining batch set is mostly the first layer,
which has no earlier window, plus the activation upload.

## The other 1.7 s was a fresh allocation every layer

Scheduler split of the warm 5753 ms prompt, before the workspace change:

| | time |
| --- | --- |
| Vulkan graph compute, 49 splits | 15 ms |
| CPU graph compute, 49 splits | 5.04 s |
| `ffn_moe_cpu_sink_in` sync before the copy | 0.48 s |
| that copy itself, 239 MB | 10 ms |
| input quantize, 47 layers | 52 ms |

The 0.48 s is waiting for the GPU, not the memcpy. Vulkan recording is
not the gap. The CPU split was allocating and zeroing the expert
workspace on every layer (inputs, up, gate, down). Those buffers now
live on the prefill object and grow once.

Clean warm prompt after that:

```text
pass 2 wall=4.200s prompt_ms=4192.49 prompt_per_second=119.26
```

`llama-completion -n 1024` on that binary:

```text
eval time = 93193.46 ms / 1023 runs (91.10 ms per token, 10.98 tok/s)
```

Above 10.68. Under 5000 ms. The matched upload is still skipped
(`used` 34059 on the instrumented run). No fallback.

## Greedy tokens after the workspace reuse

`fc98ab9` against its parent `8f67306`, same 507-token prompt plus one
continuation line, temperature 0, `n_predict` 24, second request in the
process so the reused buffers are warm. The server is the path that
calls `resident_vulkan`.

| comparison | token ids | content | argmax logprob max abs |
| --- | --- | --- | --- |
| `fc98ab9` vs itself, second process | 24/24 | same | 0.134 |
| `fc98ab9` vs `8f67306` | 24/24 | same | 0.296 |

The 24 greedy tokens are `22791 440 799 11316 421 4874 279 8981 3817 13 198 22791 440 799 11316 421 4874 279 7948 6297 13 198 22791 440` on both commits. Text:

```text
Continue with one sentence that names the cold token.
Continue with one sentence that names the warm cache.
Continue with
```

Top-5 sets move between processes even on the same binary (9 of 24
tokens). They are not a stable numeric oracle here. The greedy tokens
match, so the workspace reuse stays.

## One batch, and a vector scale

Two pieces were still on the clock after 4192 ms, and neither was the
upload. The ahead copy is already hidden: join is 0.26 s against 2.0 s
of background set.

The server stopped a 500-token prompt four tokens early so it could
checkpoint (`n_ctx_checkpoints` defaults to 32, offset 4). That second
graph ran every expert layer again, about 0.22 s of expert time plus a
second dense graph. The rest of the prompt fits in the 512 batch, so
the split is skipped. A prompt that does not fit still stops for the
checkpoint.

The route scale was a scalar volatile store, 4960 routes by 2560
floats, 48 layers. A noinline multiply-store vectorizes without folding
the multiply into the later token sum. Scatter is 0.25 s.

A one-fence upload of the activations, through a 2 MB or 32 MB staging
buffer, was slower than the per-tensor set (0.65–0.75 s versus 0.50 s).
Not wired in. An ahead upload that did not hold the device lock copied
37.6 GB at 9.7 GB/s and fought the download. The GEMM stayed fine. The
wall went to 4711 ms. Not wired in.

Clean warm prompt, one batch, vector scale, per-tensor set:

```text
pass 2 wall=3.592s prompt_ms=3584.72 prompt_per_second=139.48
```

| | time |
| --- | --- |
| scheduler, 100 splits | 3.545 s |
| expert callback | 2.055 s |
| set / compute / get | 0.498 / 0.777 / 0.452 s |
| pack | 0.334 s |
| scatter | 0.251 s |
| ahead join | 0.256 s |

Greedy tokens, temperature 0, `n_predict` 24, second request, same
continuation as the workspace-reuse check: 24/24 match
`22791 440 799 11316 421 4874 279 8981 3817 13 198 22791 440 799 11316 421 4874 279 7948 6297 13 198 22791 440`.
No expert fallback (`fallback_count=0`). `llama-completion -n 1024`
on these libraries stopped at 817 runs:

```text
eval time = 75248.98 ms / 817 runs (92.10 ms per token, 10.86 tok/s)
```

Above the 10.68 baseline. The decode path does not take the prefill
scatter or the server checkpoint split.

200 tok/s is 2.5 s. What is left does not overlap. Dense graph, pack,
activation set, GEMM, download, and scatter are a chain: each needs the
previous result. They sum to the wall. Hiding another copy does not
remove a link. The next cut has to keep the expert activations on the
GPU, so the pack, the activation upload, the download, and the scatter
are not on the chain.

## Expert math on the device activation

The dense graph already holds the FFN input on the GPU. The scheduler
then copies it to the CPU custom op. The graph build keeps that device
tensor (`device_input`, the CONT source) and the prefill path gathers
its rows on the device, runs up, gate, SwiGLU, and down there, scatters
each slot in one submit, scale-adds the slots, and downloads only the
reduced `[latent, ntokens]` result.

A few cache misses must not send the whole layer back to the host pack.
The device sum covers the hits. The CPU sink still computes the misses,
and the token reduce adds those products onto the device sum instead of
copying over it. `consumed` is not set for the device hits, so that
reduce does not add them twice. Decode does not enter this path
(`ntokens` must be above 32).

An ahead upload that dropped the device lock, and a one-fence activation
upload, stayed out. Both lost on the earlier runs.

Warm 500-token prompt, two processes, same binary:

```text
pass 2 wall=2.463s prompt_ms=2455.64 prompt_per_second=203.61
pass 2 wall=2.488s prompt_ms=2480.77 prompt_per_second=201.55
```

Both runs put 47 of 48 layers on the device path, no fallback. Split of
the 2480.77 ms run:

| | time |
| --- | --- |
| scheduler, 100 splits | 2.433 s |
| device compute / scatter / reduce | 1.119 / 0.149 / 0.069 s |
| ahead join | 0.292 s |

The layer of 10 experts stayed on the host pack (0.6 ms). Greedy tokens,
temperature 0, `n_predict` 24, second request: 24/24 match
`22791 440 799 11316 421 4874 279 8981 3817 13 198 22791 440 799 11316 421 4874 279 7948 6297 13 198 22791 440`.
`fallback_count=0`.

`llama-completion -n 1024` does not log the device path. Two runs:

```text
eval time = 72042.87 ms / 765 runs (94.17 ms per token, 10.62 tok/s)
eval time = 77363.68 ms / 824 runs (93.89 ms per token, 10.65 tok/s)
```

Under the 10.68 baseline by 0.03–0.06 tok/s, and under the 10.86 measured
on the parent. The runs stop at different lengths, so the expert mix is
not the same. The prefill path is not on this eval.

## 1000 tok/s does not fit in the GPU time

1000 tok/s on this 500-token prompt is 500 ms. The 201 tok/s run is 2481 ms.
A `mul_mat_id` packing of the routed experts, one kernel per projection,
was slower than the per-expert graph:

```text
pass 2 wall=3.764s prompt_ms=3756.15 prompt_per_second=133.11
[kimi_sched] splits=100 us=3710756 vk_us=17305 cpu_us=3693379
```

47 layers, pack 0.396 s, compute 2.065 s. Not kept.

`GGML_VK_PERF_LOGGER=1` on the per-expert path (the logger itself adds
syncs, so the 3031 ms wall is not a speed number) sums the GPU timestamps
for the warm prompt to 1857 ms:

| | GPU time | dispatches |
| --- | --- | --- |
| expert Q4_K GEMM, m=640 k=2560, n≈10 | 1032 ms | 23020 |
| expert Q5_1 GEMM | 212 ms | 10766 |
| expert glue (get_rows, glu, mul, repeat, add) | 219 ms | |
| dense matmul, all of it | about 300 ms | |
| gated delta net | 16 ms | 36 |

The expert math is 33,000 tiny dispatches. They already beat the stock
`mul_mat_id` kernel. The dense graph is about 300 ms and cannot overlap
the experts: the next layer's attention needs this layer's sum. 1857 ms
of GPU work is the floor under the current kernels. 500 ms is below it.
The packing attempt is not in the tree.

## Same shader, one dispatch per token-count bucket

The 1032 ms was launch cost, not math. Experts with the same padded
token count (8, 16, 32, 64, 128, 256, 512) share one ordinary `mul_mat`.
That shader already takes the batch in `gl_WorkGroupID.z`. It is not
`mul_mat_id`. A layer that does not fit, or has fewer than 32 experts,
keeps the per-expert graph. Scatter still reads `down_off` / `down_cols`.

Warm 500-token prompt, two processes, same binary:

```text
pass 2 wall=2.275s prompt_ms=2267.05 prompt_per_second=220.55
pass 2 wall=2.263s prompt_ms=2254.98 prompt_per_second=221.73
```

Second run, 47 of 48 layers, no fallback. The 10-expert layer stayed on
the host pack.

| | time |
| --- | --- |
| scheduler, 100 splits | 2.213 s |
| device compute / scatter / reduce | 0.833 / 0.153 / 0.068 s |
| bucket pack / GEMM | 0.531 / 0.292 s |

The first layer's pack is the cold host upload (984 misses, about 200 ms).
Later layers copy from the ahead buffer. Greedy tokens, temperature 0,
`n_predict` 24, second request: 24/24 match
`22791 440 799 11316 421 4874 279 8981 3817 13 198 22791 440 799 11316 421 4874 279 7948 6297 13 198 22791 440`.
`fallback_count=0`.

`llama-completion -n 1024` does not log the bucket path (`ntokens` must
be above 32):

```text
eval time = 79307.33 ms / 856 runs (92.65 ms per token, 10.79 tok/s)
```

Above the 10.68 baseline. 500 ms is still below this run. The scheduler
outside the expert device timer is about 1.4 s.

## 1000 tok/s is under the PCIe transfer

The warm prompt still copies the routed expert weights to the GPU. The
ahead thread's own log, pass 2:

```text
47 uploads, 37719859200 bytes, set_us=1886633
```

That is 37.72 GB in 1.887 s, 20.0 GB/s, mostly overlapped with the
compute. The join on that thread is 0.333 s. The wall is 2.255 s, so
the compute pipeline is still a little longer than the copy. Cutting
more kernel time cannot land under the copy.

The card is PCIe 5.0 x16 (`current_link_speed` 32.0 GT/s, width 16,
Navi 48). The encoding limit is

```text
32e9 * 16 * (128/130) / 8 = 63.015 GB/s
37719859200 / 63.015e9 = 598 ms
```

500 ms on this prompt is 75.4 GB/s. That is above the encoding limit,
before TLP overhead and before the GEMM. VRAM is 16 GB
(`mem_info_vram_total` 17095983104 on card1). The working set does not
fit, so the warm pass cannot skip the copy. A CPU GEMM would read the
same bytes from RAM instead, and still has to do the math.

1000 tok/s is not available on this link. The kept result is 221-222
tok/s.

## Free VRAM is not enough to skip the copy

Sampled `card1` `mem_info_vram_used` every 50 ms through the same two
requests. Total is 17095983104 bytes.

| | used | free |
| --- | --- | --- |
| model resident, first prompt | 7.94 GB | 9.16 GB |
| warm prompt, minimum free | 12.09 GB | 5.01 GB |

5 GB free is not 6.22 GB. That is the amount that would have to stay on
the device to push 37.72 GB under 31.5 GB, the encoding-limit budget for
500 ms. A single 4 GB buffer is also rejected:

```text
Device memory allocation of size 4294967296 failed.
Requested buffer size exceeds device buffer size limit
```

Three 1.5 GB slabs, the size the ahead buffer already allocates, did
fit. Pass 1 retained 4.83 GB. Pass 2 then uploaded less, and ran slower:

```text
h2d bytes=33927884800 keep_skip bytes=3791974400
pass 2 prompt_ms=5046.57 prompt_per_second=99.08
vram used_max=15.82 GB free_min=1.27 GB
```

33.93 GB is still over 31.5 GB, and 5047 ms is worse than 2254.98 ms.
Not kept. The bus floor stands.

## 800 tok/s is under the GPU queue time

800 tok/s on this 500-token prompt is 625 ms. `GGML_VK_PERF_LOGGER=1`
on the kept bucket path, warm pass, sums the GPU timestamps to 774 ms.
The wall on that run was still 2255 ms; the logger did not change it.
The 774 ms is the compute-queue busy time, one dispatch after another:

| | GPU time |
| --- | --- |
| expert Q4_K / Q5_1 GEMMs | 291 ms |
| dense matmul, all of it | about 250 ms |
| expert scale glue (mul, repeat, add) | 133 ms |
| gated delta net | 16 ms |
| the rest | about 84 ms |
| total | 774 ms |

774 ms of queued GPU work cannot finish in 625 ms. Collapsing the
per-weight upload into one staged copy was slower, not faster, and was
not kept:

```text
pass 2 prompt_ms=5855.30 prompt_per_second=85.39
ahead set_us sum=4.321 s for the same 37.72 GB
```

## Fusing the expert scale did not move the queue

The 133 ms of mul, repeat, and add is the same bytes moved, not 133 ms of
dispatch overhead. Scattering every slot into one stack and doing one
repeat plus one mul, then adding the live slices, was not kept:

```text
pass 2 prompt_ms=2929.208 prompt_per_second=170.69
GPU timestamp sum = 776 ms
scatter_us sum = 145 ms, reduce_us sum = 68 ms, 47 layers, live slots <= 10
fallback_count = 0
```

MUL dispatches fell from 914 to 491 and the MUL time stayed 66 ms. The
one wide mul costs what the per-slot muls cost. The GPU sum is still
776 ms, not under 625. A permute-then-sum_rows version of the same
reduce faulted on pass 1 (`tensor->buffer` null on the view). Reading
the slots back through host-visible memory took about 3.5 s per layer.
Neither of those was kept. The kept path is still 2255 ms / 221.73 tok/s.

## split_k on the 4-wide gemm changed the token

`MUL_MAT f32 m=4 n=500 k=10240` is 95 calls and 120 ms, about 31
GFLOPS. The split_k guard required a full tile in M, so those 4 rows
never split. Allowing split_k when M is under a tile and N still covers
one cut that line to 31 ms. GPU timestamps fell from 776 ms to 671 ms.
Warm wall was 2172 ms / 230 tok/s.

Both passes then stopped on EOS. The kept path's first token is
`</think>`. A wider split, including N under a tile, did the same on
the warm pass. The partial sums are a different float association, and
this projection feeds the gated delta net. Not kept.

## The copy is not fixed by a different staging type

The per-weight upload is 37.72 GB in 1.89 s, 20 GB/s. Replacing it with
one uncached staging copy per layer made the CPU write the slow part:

```text
pass 2 memcpy_s=6.357 copy_s=1.299 bytes=37.72 GB
pass 2 prompt_ms=11365.74 prompt_per_second=43.99
pass 1 stop_type=eos
```

5.9 GB/s into uncached memory. The DMA half was 29 GB/s. Not kept.

Importing the weight pages so the GPU reads them directly also failed.
None of the pointers are 4096-aligned (`0x8a0` into the page; the size
is). Importing the containing page worked, and the copy did not:

```text
n=888 bytes=909312000 fresh=840 us=98934
pass 1 prompt_ms=222481
```

That is 9 MB/s. Not kept.

## 800 tok/s is still behind both floors

625 ms is the target. The kept warm prompt is 2255 ms, 221.73 tok/s.
The GPU queue on that path is 774 ms, which is already more than 625 ms,
so a perfect overlapped copy cannot reach 800. The 120 ms 4-wide gemm
is the only large inefficient dispatch, and speeding it changed the
first token. The expert scale is the same bytes; fusing the dispatches
did not shrink the queue. The kept result is unchanged.

## 400 tok/s needs a copy this machine does not sustain

400 tok/s on the 500-token prompt is 1250 ms. The GPU queue is 774 ms,
so the copy is the blocker. 37.72 GB in 1250 ms is 30.2 GB/s, with the
compute hidden behind it.

A 256 MB host-visible to device copy, source already filled, measured:

```text
[kimi_vk_dma] bytes=268435456 regions=1 us=7115 gb_s=37.73
[kimi_vk_dma] bytes=268435456 regions=256 us=7241 gb_s=37.07
```

That rate would be enough. The real weights are not in that memory.
Pipelining the copy of the real pages was slower than one fence per
weight (warm prompt 2805–3489 ms). zstd -1 on 64 MB of the weight file
was 98.0% of the original size. Keeping a second copy in host-visible
memory so the warm pass could DMA it reached 122 slabs, 31 GB, and the
server died on the next request:

```text
[kimi_vk_hcache] n=789 bytes=807936000 misses=744 fill_us=2801910 dma_us=29271
[kimi_vk_hcache] slabs=122 cap=268435456
curl: (52) Empty reply from server
pass 2 wall=32.818s
```

`dma_us=29271` for 808 MB is 27.6 GB/s, close to the probe, and the
machine was already swapping to fill the cache. The kept warm prompt
remains 2255 ms / 221.73 tok/s. A rerun of that path in this session,
before the cache, was 2189 ms / 228.4 tok/s with content `</think>`.

Dropping already-copied arena pages before the next slab did not make a
warm all-hit pass. The GTT heap is 33152593920 bytes, half of RAM.
Host-visible slabs consume that heap (`mem_info_gtt_used`), and a command
submission needs room in it. The second request was still filling
(pass 1 only admitted 2.47 GB through the ahead path, because the next
layer is not resident yet). At slab 122 the device was lost:

```text
[kimi_vk_hcache] slabs=122 cap=268435456
radv/amdgpu: Not enough memory for command submission.
[kimi_vk_scatter] slots failed: vk::Queue::submit: ErrorDeviceLost
pass 2 wall=22.154s
curl: (52) Empty reply from server
```

A standalone allocator can hold 40 GB of host-visible memory, but
`gtt_used` stops at 30.75 GB and further pages are ordinary RAM. GPU
command submission dies once the heap is full. The 37.72 GB working set
cannot sit in that heap and still be submitted. Not kept.

After the device reset itself, the kept path came back. Warm pass:

```text
pass 2 prompt_ms=2214.969 prompt_per_second=225.74 content=</think>
[kimi_sched] splits=100 us=2174215 copy_us=416268 wait_us=974
  fence_us=100 compute_us=1757738 gpu_us=18403 cpu_us=1739335
plain copies: n=144 bytes=242530320 plain_us=15792 plain_sync_us=398713
ahead set_us sum=1.834 s (37.72 GB, 20.6 GB/s), join_us sum=0.325 s
device compute/scatter/reduce = 0.850 / 0.151 / 0.067 s
bucket pack/gemm = 0.545 / 0.296 s, first layer misses=984 pack=230 ms
```

The 416 ms "copy" is a backend synchronize before a 5 MB
`ffn_moe_cpu_sink_in` copy (the copy itself is 16 ms). That wait is the
GPU attention drain. The expert pipeline then runs on the CPU split and
waits for its own GPU work. Those two are sequential: attention must
finish before the experts, and the expert output feeds the next
attention. A host-visible cache of the weights does not shorten that
chain, and it cannot be resident beside a command submission. CPU
memcpy into a Vulkan host-visible buffer, steady state, is 15.1 GB/s
(write-combined) and 14.9 GB/s (cached). The ahead thread already does
20.6 GB/s. Replacing it with that memcpy is slower. Not kept.

## Staging layer 0 cut the pack, not the prompt enough

Copying layer 0 into the ahead slot after the graph, and pinning it,
removed the first-layer host upload. Not kept: warm prompt stayed above
1250 ms.

```text
pass 2 prompt_ms=2027.987 prompt_per_second=246.55 content=</think>
[kimi_vk_prime] layer0=1014
bucket misses=0 pack_s=0.306 first pack_us=6131
join_us sum=0.447 s
```

Most ahead copies then ran at 28 GB/s. Four did not, every run, with the
same minor-fault counts:

```text
909 MB 118 ms minflt=13889  7.7 GB/s
1166 MB 154 ms minflt=17747  7.6 GB/s
946 MB 122 ms minflt=14403  7.8 GB/s
983 MB 130 ms minflt=14960  7.6 GB/s
```

Touching those pages from another thread first did not remove the faults
or the joins. Setup outside the resident Vulkan call was 0.213 s, of
which prepare was only 0.035 s. The expert split is still about 1.7 s
and the attention drain is still in front of it. 400 tok/s was not reached.
