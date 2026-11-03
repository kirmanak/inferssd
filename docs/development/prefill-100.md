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
   slot, publish after the consumer. The next prompt's existing
   `prepare_prefill_read` should hit. Cold path still streams, so the first
   prompt should stay near 59 tok/s.
2. When a layer has any cache hit, wait out the remaining misses and run one
   Vulkan graph for the up/gate pairs and one for the down projections.
   A fully cold layer stays on the streaming consumer so compute still
   overlaps the read. `INFERSSD_PREFILL_GPU=0` disables the batch.

Not tried yet, in order if the warm rate is short of 100:

- Overlap the miss reads with the hit batch, instead of waiting first.
- Import the cache arena instead of staging a copy.
- Sort cold reads by file offset. That can approach 7 GB/s, about 70 tok/s
  cold, and does not reach 100.
- A fused SwiGLU so the down projection does not wait on a CPU round trip.

## Results

Not measured yet. The number that counts is the second prompt in one
process, same 507-token file, greedy, with decode tok/s on a short
generation still near 10 tok/s once the cache is warm.
