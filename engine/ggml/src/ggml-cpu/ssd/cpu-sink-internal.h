#pragma once

#include "cpu-sink.h"
#include "ggml-ssd/gpu/vulkan-expert.h"
#include <cmath>
#include "ggml-ssd/telemetry/telemetry.h"

// Matches mul(weight) followed by ggml_rms_norm, which squares in float and sums in double, then route scaling.
inline void kimi_cpu_sink_normalize_expert(float * values, int rows, const float * norm, float eps) {
    double sum = 0.0;
    for (int row = 0; row < rows; ++row) {
        values[row] *= norm[row];
        sum += double(values[row] * values[row]);
    }
    const float scale = 1.0f / std::sqrt(float(sum / rows) + eps);
    for (int row = 0; row < rows; ++row) { values[row] *= scale; }
}


// Private executor hooks. The stream serializes all accesses to the shared
// decode ring and compute phases. Call parallel only from the stream consumer,
// and parallel_last only from the caller of a stream that returned, as the
// open call's last phase; callbacks must not recursively enter either executor
// or allocate/throw.
// Streaming is a prefill operation: it disables decode training. A retained
// arena may still be read, and a miss may be admitted into a resident slot so
// the next prompt hits. resident_vulkan, when set, replaces the per-expert
// consumer for a layer whose experts are all cache hits.
using kimi_cpu_sink_parallel_fn = void (*)(void * userdata, int ith, int nth);
using kimi_cpu_sink_consumer_fn = void (*)(void * userdata, size_t expert,
        const void * up, const void * gate, const void * down);

struct kimi_cpu_sink_prefill_batch {
    const void * quantized_input;
    size_t quantized_stride;
    const float * f32_input;
    size_t f32_stride;
    const int32_t * route_request_index;
    const float * weights;
    int nslots;
    int ntokens;
    struct kimi_cpu_sink_activation activation;
    float * output;
    uint64_t gpu_experts;
    uint64_t gpu_routes;
    uint64_t gpu_transient_experts;
    uint64_t gpu_transient_routes;
    uint64_t gpu_transient_upload_bytes;
    double gpu_transient_upload_us;
    uint64_t gpu_ssd_transient_candidates;
    uint64_t gpu_ssd_transient_experts;
    uint64_t gpu_ssd_transient_routes;
    uint64_t gpu_ssd_transient_upload_bytes;
    double gpu_ssd_transient_upload_us;
    double gpu_worker_us;
    double gpu_expert_us;    // MXFP4 batch path: the GPU worker's time in its expert computations
    double gpu_read_wait_us; // the GPU worker's wait for streamed SSD reads
    uint64_t gpu_cold_admit_candidates;
    uint64_t gpu_cold_admit_experts;
    uint64_t gpu_cold_admit_routes;
    uint64_t gpu_cold_admit_upload_bytes;
    double gpu_cold_admit_upload_us;
    uint64_t ram_experts;
    uint64_t ram_routes;
    uint64_t ram_bytes;
    uint64_t ssd_experts;
    uint64_t ssd_routes;
    uint64_t ssd_bytes;
    // Payload pointers, data_offset already applied, request order. Return false
    // before writing outputs to fall back to the per-expert consumer.
    bool (*resident_vulkan)(void * userdata, const void * const * up, const void * const * gate,
            const void * const * down, int nexperts);
    // Optional. A miss read into a reserved host-cache slot is offered to
    // defer_vulkan instead of the consumer. True means the consumer took it for
    // its device worker, which reads the payload until flush_vulkan returns; the
    // slot stays unpublished until then. flush_vulkan waits for that worker and
    // is false if any taken expert is still undone: the stream then runs the
    // consumer for every one it deferred, and the consumer skips finished ones.
    bool (*defer_vulkan)(void * userdata, size_t expert, const void * up, const void * gate, const void * down);
    bool (*flush_vulkan)(void * userdata);
    // Fewer candidates than this in a layer are not offered: a short batch run
    // after the last read costs more than computing them as they arrive.
    size_t defer_min;
};

void kimi_cpu_sink_parallel(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata);
void kimi_cpu_sink_parallel_last(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata);
void kimi_cpu_sink_vulkan_begin(struct kimi_cpu_sink * sink, uint64_t budget);
void kimi_cpu_sink_vulkan_result(struct kimi_cpu_sink * sink, int gpu, int32_t expert,
        uint64_t bytes, double microseconds, uint64_t routes);
const struct ggml_vk_expert_api_v1 * kimi_cpu_sink_vulkan_api(const struct kimi_cpu_sink * sink);
void * kimi_cpu_sink_vulkan_session(struct kimi_cpu_sink * sink);
uint64_t kimi_cpu_sink_vulkan_budget(const struct kimi_cpu_sink * sink);

bool kimi_cpu_sink_stream_hybrid(struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        struct kimi_cpu_sink_prefill_batch * batch,
        kimi_cpu_sink_consumer_fn consumer, void * userdata,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size);

// Borrowed immutable selection; the prefill owner never outlives its sink.
const ssd::telemetry::debug_channels & kimi_cpu_sink_debug(const kimi_cpu_sink * sink) noexcept;
