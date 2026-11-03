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
