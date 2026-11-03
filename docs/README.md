<!-- SPDX-License-Identifier: Apache-2.0 -->
# Documentation

[Back to project README](../README.md)

## Using InferSSD

- [Quick start](quickstart.md): build and launch on macOS or Linux.
- [Qwen3.8-Flash-Next on Vulkan](qwen38-vulkan.md): the local RX 9070 XT changes, measured rates, and the OpenCode server.
- [Apple Silicon deployment](mac-launcher.md): Metal build, cache and thread settings.
- [Configuration](configuration.md): the typed `inferssd-settings 1` contract.
- [Deployment resources](engine-resources.md): engine descriptors for installations outside a checkout.
- [Machine profile](machine-profile.md): what `inferssd profile` measures and where it keeps the result.
- [Framework commands](../framework/README.md): source preparation, planning and owned launch.

## Design

- [How it works](architecture.md): expert sources, caches, batching, layering and the module map.
- [Controller observations](architecture/controller-observations.md): native events consumed by the Linux controller.
- [Runtime telemetry](telemetry.md): the versioned JSONL writer and reader.
- [Typed reader configuration](ssd-reader-configuration.md): reader settings, ownership and fixtures.
- Architecture decisions: [owned runtime](adr/0001-owned-runtime.md), [I/O boundaries](adr/0002-io-boundaries.md), [resource admission](adr/0003-resource-admission.md) and [configuration and evidence](adr/0004-configuration-and-evidence.md).

## Benchmarks and testing

- [Benchmark methodology](benchmark.md): hardware, definitions and archived run records.
- [Current 4K comparison data](benchmarks/current-comparison.json): rates and speedups matching the README.
- [Benchmark tools](../bench/README.md): measurement drivers and result contracts.
- [Test gates](testing.md): unit, device and Apple Silicon test coverage.

## Development

- [Upstream patches](development/upstream-patches.md): integration points in the bundled llama.cpp snapshot.
- [GPU prefill overhead](development/prefill-overhead.md): the placement cost model for batched GPU work.

