// Prompt-expert matmul on the Vulkan backend's existing quantized kernels.
// One staging buffer is reused across experts. No new GEMM shader.
#include "../ggml-ssd/gpu/vulkan-expert.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

namespace {

constexpr uint64_t k_staging_cap = 64ull << 20;

void set_error(char * error, size_t error_size, const char * message) {
    if (error && error_size) { std::snprintf(error, error_size, "%s", message ? message : ""); }
}

bool expert_supports(int32_t type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

struct expert_session {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t staging = 0;
    ggml_backend_buffer_t batch_buffer = nullptr;
    ggml_backend_buffer_type_t batch_buft = nullptr;
    uint64_t batch_staging = 0;
    ggml_context * ctx = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    int32_t type = -1;
    int64_t k = 0;
    int64_t n = 0;
    int n_tokens = 0;

    explicit expert_session(ggml_backend_t device) : backend(device) {}

    void drop_graph() {
        if (ctx) { ggml_free(ctx); }
        ctx = nullptr;
        weight = input = output = nullptr;
        graph = nullptr;
        type = -1;
        k = n = 0;
        n_tokens = 0;
    }

    ~expert_session() {
        drop_graph();
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (batch_buffer) { ggml_backend_buffer_free(batch_buffer); }
    }
};

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool expert_prepare(expert_session * session, int32_t type, int64_t k, int64_t n, int n_tokens, char * error, size_t error_size) {
    if (session->ctx && session->type == type && session->k == k && session->n == n && session->n_tokens == n_tokens) {
        return true;
    }
    session->drop_graph();
    if (k <= 0 || n <= 0 || n_tokens <= 0) {
        set_error(error, error_size, "vulkan expert matmul shape is empty");
        return false;
    }
    const int block = ggml_blck_size(ggml_type(type));
    if (block <= 0 || k % block != 0) {
        set_error(error, error_size, "vulkan expert K is not a whole number of quant blocks");
        return false;
    }
    const size_t mem = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 4096;
    session->ctx = ggml_init({ mem, nullptr, true });
    if (!session->ctx) {
        set_error(error, error_size, "vulkan expert context failed");
        return false;
    }
    session->weight = ggml_new_tensor_2d(session->ctx, ggml_type(type), k, n);
    session->input = ggml_new_tensor_2d(session->ctx, GGML_TYPE_F32, k, n_tokens);
    session->output = session->weight && session->input ? ggml_mul_mat(session->ctx, session->weight, session->input) : nullptr;
    if (!session->output) {
        session->drop_graph();
        set_error(error, error_size, "vulkan expert graph build failed");
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session->backend);
    const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), 128);
    ggml_tensor * tensors[3] = { session->weight, session->input, session->output };
    size_t cursor = 0;
    size_t offsets[3] = {};
    for (int i = 0; i < 3; ++i) {
        cursor = align_up(cursor, alignment);
        offsets[i] = cursor;
        cursor += ggml_backend_buft_get_alloc_size(buft, tensors[i]);
    }
    if (cursor == 0 || cursor > k_staging_cap) {
        session->drop_graph();
        set_error(error, error_size, "vulkan expert staging exceeds the reused buffer cap");
        return false;
    }
    if (!session->buffer || session->staging < cursor) {
        if (session->buffer) {
            ggml_backend_buffer_free(session->buffer);
            session->buffer = nullptr;
            session->staging = 0;
        }
        session->buffer = ggml_backend_buft_alloc_buffer(buft, cursor);
        if (!session->buffer) {
            session->drop_graph();
            set_error(error, error_size, "vulkan expert staging allocation failed");
            return false;
        }
        session->staging = ggml_backend_buffer_get_size(session->buffer);
    }
    void * base = ggml_backend_buffer_get_base(session->buffer);
    for (int i = 0; i < 3; ++i) {
        if (ggml_backend_tensor_alloc(session->buffer, tensors[i], static_cast<char *>(base) + offsets[i]) != GGML_STATUS_SUCCESS) {
            session->drop_graph();
            set_error(error, error_size, "vulkan expert tensor placement failed");
            return false;
        }
    }
    session->graph = ggml_new_graph_custom(session->ctx, 16, false);
    if (!session->graph) {
        session->drop_graph();
        set_error(error, error_size, "vulkan expert graph allocation failed");
        return false;
    }
    ggml_build_forward_expand(session->graph, session->output);
    session->type = type;
    session->k = k;
    session->n = n;
    session->n_tokens = n_tokens;
    return true;
}

void * expert_session_create(ggml_backend_t backend, char * error, size_t error_size) {
    if (!backend) {
        set_error(error, error_size, "missing vulkan backend");
        return nullptr;
    }
    return new (std::nothrow) expert_session(backend);
}

void expert_session_destroy(void * session) {
    delete static_cast<expert_session *>(session);
}

uint64_t expert_staging_bytes(void * session) {
    const auto * value = static_cast<const expert_session *>(session);
    return value ? value->staging : 0;
}

bool expert_mul_mat(void * raw, int32_t type, int64_t k, int64_t n, int n_tokens,
        const void * weight, size_t weight_bytes, const float * input, float * output,
        char * error, size_t error_size) {
    auto * session = static_cast<expert_session *>(raw);
    if (!session || !weight || !input || !output) {
        set_error(error, error_size, "vulkan expert matmul is missing buffers");
        return false;
    }
    if (!expert_supports(type)) {
        set_error(error, error_size, "vulkan expert type has no existing matmul pipeline");
        return false;
    }
    if (!expert_prepare(session, type, k, n, n_tokens, error, error_size)) { return false; }
    if (ggml_nbytes(session->weight) != weight_bytes) {
        set_error(error, error_size, "vulkan expert weight byte count mismatch");
        return false;
    }
    ggml_backend_tensor_set(session->weight, weight, 0, weight_bytes);
    ggml_backend_tensor_set(session->input, input, 0, size_t(k) * size_t(n_tokens) * sizeof(float));
    const enum ggml_status status = ggml_backend_graph_compute(session->backend, session->graph);
    if (status != GGML_STATUS_SUCCESS) {
        set_error(error, error_size, "vulkan expert graph compute failed");
        return false;
    }
    ggml_backend_synchronize(session->backend);
    ggml_backend_tensor_get(session->output, output, 0, size_t(n) * size_t(n_tokens) * sizeof(float));
    return true;
}

constexpr uint64_t k_batch_cap = 1536ull << 20;
constexpr int k_batch_experts = 512;

bool expert_batch_chunk(expert_session * session, const ggml_vk_expert_matmul * items, int begin, int count,
        char * error, size_t error_size) {
    if (count <= 0) { return true; }
    const int graph_nodes = count * 8 + 64;
    const size_t mem = ggml_tensor_overhead() * size_t(count) * 4 + ggml_graph_overhead_custom(graph_nodes, false) + 65536;
    ggml_context * ctx = ggml_init({ mem, nullptr, true });
    if (!ctx) {
        set_error(error, error_size, "vulkan expert batch context failed");
        return false;
    }
    struct item_tensors {
        ggml_tensor * weight = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
    };
    const size_t n_items = size_t(count);
    std::vector<item_tensors> built(n_items);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session->backend);
    const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), 128);
    size_t cursor = 0;
    std::vector<size_t> offsets(size_t(count) * 3, 0);
    for (int i = 0; i < count; ++i) {
        const auto & item = items[begin + i];
        if (!expert_supports(item.type) || item.k <= 0 || item.n <= 0 || item.n_tokens <= 0 ||
                !item.weight || !item.input || !item.output) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan expert batch item is incomplete");
            return false;
        }
        const int block = ggml_blck_size(ggml_type(item.type));
        if (block <= 0 || item.k % block != 0) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan expert batch K is not a whole number of quant blocks");
            return false;
        }
        built[size_t(i)].weight = ggml_new_tensor_2d(ctx, ggml_type(item.type), item.k, item.n);
        built[size_t(i)].input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, item.k, item.n_tokens);
        built[size_t(i)].output = built[size_t(i)].weight && built[size_t(i)].input ?
            ggml_mul_mat(ctx, built[size_t(i)].weight, built[size_t(i)].input) : nullptr;
        if (!built[size_t(i)].output) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan expert batch graph build failed");
            return false;
        }
        if (ggml_nbytes(built[size_t(i)].weight) != item.weight_bytes) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan expert batch weight byte count mismatch");
            return false;
        }
        ggml_tensor * tensors[3] = { built[size_t(i)].weight, built[size_t(i)].input, built[size_t(i)].output };
        for (int t = 0; t < 3; ++t) {
            cursor = align_up(cursor, alignment);
            offsets[size_t(i) * 3 + t] = cursor;
            cursor += ggml_backend_buft_get_alloc_size(buft, tensors[t]);
        }
    }
    if (cursor == 0 || cursor > k_batch_cap) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan expert batch exceeds the staging cap");
        return false;
    }
    if (!session->batch_buffer || session->batch_staging < cursor || session->batch_buft != buft) {
        if (session->batch_buffer) {
            ggml_backend_buffer_free(session->batch_buffer);
            session->batch_buffer = nullptr;
            session->batch_staging = 0;
        }
        session->batch_buffer = ggml_backend_buft_alloc_buffer(buft, cursor);
        session->batch_buft = buft;
        if (!session->batch_buffer) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan expert batch staging allocation failed");
            return false;
        }
        session->batch_staging = ggml_backend_buffer_get_size(session->batch_buffer);
    }
    void * base = ggml_backend_buffer_get_base(session->batch_buffer);
    for (int i = 0; i < count; ++i) {
        ggml_tensor * tensors[3] = { built[size_t(i)].weight, built[size_t(i)].input, built[size_t(i)].output };
        for (int t = 0; t < 3; ++t) {
            if (ggml_backend_tensor_alloc(session->batch_buffer, tensors[t],
                    static_cast<char *>(base) + offsets[size_t(i) * 3 + t]) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx);
                set_error(error, error_size, "vulkan expert batch tensor placement failed");
                return false;
            }
        }
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan expert batch graph allocation failed");
        return false;
    }
    for (int i = 0; i < count; ++i) { ggml_build_forward_expand(graph, built[size_t(i)].output); }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        const auto & item = items[begin + i];
        ggml_backend_tensor_set(built[size_t(i)].weight, item.weight, 0, item.weight_bytes);
        ggml_backend_tensor_set(built[size_t(i)].input, item.input, 0,
                size_t(item.k) * size_t(item.n_tokens) * sizeof(float));
    }
    const auto set_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const enum ggml_status status = ggml_backend_graph_compute(session->backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan expert batch graph compute failed");
        return false;
    }
    ggml_backend_synchronize(session->backend);
    const auto compute_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() - set_us;
    const auto t1 = std::chrono::steady_clock::now();
    const size_t n_out = size_t(count);
    std::vector<const ggml_tensor *> outs(n_out);
    std::vector<void *> dsts(n_out);
    std::vector<size_t> sizes(n_out);
    for (int i = 0; i < count; ++i) {
        const auto & item = items[begin + i];
        outs[size_t(i)] = built[size_t(i)].output;
        dsts[size_t(i)] = item.output;
        sizes[size_t(i)] = size_t(item.n) * size_t(item.n_tokens) * sizeof(float);
    }
    if (!ggml_vk_copy_tensors_fenced(outs.data(), dsts.data(), sizes.data(), count)) {
        for (int i = 0; i < count; ++i) {
            ggml_backend_tensor_get(built[size_t(i)].output, dsts[size_t(i)], 0, sizes[size_t(i)]);
        }
    }
    const auto get_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
    if (count >= 64) {
        std::fprintf(stderr, "[kimi_vulkan_batch] n=%d set_us=%.0f compute_us=%.0f get_us=%.0f\n",
                count, set_us, compute_us, get_us);
    }
    ggml_free(ctx);
    return true;
}

bool expert_mul_mat_batch(void * raw, const ggml_vk_expert_matmul * items, int count,
        char * error, size_t error_size) {
    auto * session = static_cast<expert_session *>(raw);
    if (!session || (count > 0 && !items)) {
        set_error(error, error_size, "vulkan expert batch is missing buffers");
        return false;
    }
    if (count < 0) {
        set_error(error, error_size, "vulkan expert batch count is negative");
        return false;
    }
    int begin = 0;
    while (begin < count) {
        uint64_t bytes = 0;
        int n = 0;
        while (begin + n < count && n < k_batch_experts) {
            const auto & item = items[begin + n];
            const uint64_t next = uint64_t(item.weight_bytes) +
                uint64_t(item.k) * uint64_t(item.n_tokens) * sizeof(float) +
                uint64_t(item.n) * uint64_t(item.n_tokens) * sizeof(float);
            if (n && bytes + next > k_batch_cap) { break; }
            bytes += next;
            ++n;
        }
        if (!expert_batch_chunk(session, items, begin, n, error, error_size)) { return false; }
        begin += n;
    }
    return true;
}

}  // namespace

extern "C" const ggml_vk_expert_api_v1 * ggml_backend_vk_expert_api_v1(void) {
    static const ggml_vk_expert_api_v1 api = {
        1,
        uint32_t(sizeof(ggml_vk_expert_api_v1)),
        reinterpret_cast<bool (*)(int32_t)>(expert_supports),
        reinterpret_cast<void * (*)(ggml_backend_t, char *, size_t)>(expert_session_create),
        reinterpret_cast<void (*)(void *)>(expert_session_destroy),
        reinterpret_cast<uint64_t (*)(void *)>(expert_staging_bytes),
        reinterpret_cast<bool (*)(void *, int32_t, int64_t, int64_t, int, const void *, size_t, const float *, float *, char *, size_t)>(expert_mul_mat),
        reinterpret_cast<bool (*)(void *, const ggml_vk_expert_matmul *, int, char *, size_t)>(expert_mul_mat_batch),
    };
    return &api;
}
