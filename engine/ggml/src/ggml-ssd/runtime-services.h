// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace ssd {
namespace store { struct source_view; }
namespace gpu { struct provider_v1; struct cpu_kernels_v1; }
namespace host { class context_client; }
namespace telemetry {
class writer;
struct debug_channels;
}

// The runtime owns these services. Executors copy the debug selection and borrow
// the writer only until their workers and publishers have all joined.
struct runtime_services {
    telemetry::writer * events = nullptr;
    const telemetry::debug_channels * debug = nullptr;
    host::context_client * host = nullptr;
    const gpu::provider_v1 * gpu = nullptr;
    const gpu::cpu_kernels_v1 * cpu = nullptr;
    int gpu_device = 0;
    const store::source_view * source = nullptr;
    void * gpu_transport = nullptr;
    size_t gpu_reserve_bytes = 0;
    // Vulkan prompt experts (ggml_vk_expert_api_v1). Absent on CUDA and CPU.
    // vulkan_free_bytes is device memory free after the dense model, so a budget
    // cannot evict that buffer. Not a CUDA provider and not read by tier_bytes().
    const void * vulkan_experts = nullptr;
    void * vulkan_backend = nullptr;
    uint64_t vulkan_free_bytes = 0;
};
}  // namespace ssd
