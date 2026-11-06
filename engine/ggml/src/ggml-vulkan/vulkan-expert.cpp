// Prompt-expert matmul on the Vulkan backend's existing quantized kernels.
// One staging buffer is reused across experts. No new GEMM shader.
#include "../ggml-ssd/gpu/vulkan-expert.h"
#include "ggml-vulkan.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct decode_cache;
std::atomic<decode_cache *> live_cache{nullptr};
std::atomic<int> upload_window_flag{0};

constexpr uint64_t k_staging_cap = 64ull << 20;
// Set while a decode layer walks misses. Uploads copy about 4 MB each and
// were stealing that bandwidth from the walk.
std::atomic<int> decode_walk_active{0};

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

struct decode_cache;

struct expert_session {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t staging = 0;
    ggml_backend_buffer_t batch_buffer = nullptr;
    ggml_backend_buffer_type_t batch_buft = nullptr;
    uint64_t batch_staging = 0;
    struct ahead_slot {
        ggml_backend_buffer_t buffer = nullptr;
        uint64_t cap = 0;
        std::unordered_map<const void *, size_t> at;
        std::unordered_map<const void *, size_t> bytes;
    };
    ahead_slot ahead_slots[2];
    int ahead_shown = 0;
    int ahead_filling = -1;
    ggml_backend_buffer_t ahead_buffer = nullptr;
    std::thread ahead_thread;
    std::unordered_map<const void *, size_t> ahead_at;
    std::unordered_map<const void *, size_t> ahead_bytes;
    int ahead_used = 0;
    ggml_backend_buffer_t slot_buf[16] = {};
    uint64_t slot_bytes = 0;
    ggml_context * ctx = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    int32_t type = -1;
    int64_t k = 0;
    int64_t n = 0;
    int n_tokens = 0;
    decode_cache * decode = nullptr;

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

    void join_ahead() {
        if (!ahead_thread.joinable()) { return; }
        const auto t0 = std::chrono::steady_clock::now();
        ahead_thread.join();
        const double join_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[kimi_vk_ahead] join_us=%.0f\n", join_us);
        if (ahead_filling >= 0) {
            ahead_shown = ahead_filling;
            ahead_filling = -1;
            ahead_buffer = ahead_slots[ahead_shown].buffer;
            ahead_at.swap(ahead_slots[ahead_shown].at);
            ahead_bytes.swap(ahead_slots[ahead_shown].bytes);
        }
    }

    ~expert_session();
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
    const auto t_enter = std::chrono::steady_clock::now();
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
    std::vector<size_t> ahead_off(size_t(count), SIZE_MAX);
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
        const auto ahead = session->ahead_at.find(item.weight);
        if (ahead != session->ahead_at.end() && session->ahead_buffer &&
                session->ahead_bytes[item.weight] >= item.weight_bytes &&
                ahead->second + item.weight_bytes <= ggml_backend_buffer_get_size(session->ahead_buffer)) {
            ahead_off[size_t(i)] = ahead->second;
        }
        for (int t = 0; t < 3; ++t) {
            if (t == 0 && ahead_off[size_t(i)] != SIZE_MAX) { continue; }
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
    void * ahead_base = session->ahead_buffer ? ggml_backend_buffer_get_base(session->ahead_buffer) : nullptr;
    for (int i = 0; i < count; ++i) {
        ggml_tensor * tensors[3] = { built[size_t(i)].weight, built[size_t(i)].input, built[size_t(i)].output };
        if (ahead_off[size_t(i)] != SIZE_MAX) {
            if (ggml_backend_tensor_alloc(session->ahead_buffer, tensors[0],
                    static_cast<char *>(ahead_base) + ahead_off[size_t(i)]) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx);
                set_error(error, error_size, "vulkan expert ahead placement failed");
                return false;
            }
        }
        for (int t = 0; t < 3; ++t) {
            if (t == 0 && ahead_off[size_t(i)] != SIZE_MAX) { continue; }
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
    const auto build_us = std::chrono::duration<double, std::micro>(t0 - t_enter).count();
    for (int i = 0; i < count; ++i) {
        const auto & item = items[begin + i];
        if (ahead_off[size_t(i)] == SIZE_MAX) {
            ggml_backend_tensor_set(built[size_t(i)].weight, item.weight, 0, item.weight_bytes);
        } else {
            ++session->ahead_used;
        }
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
        std::fprintf(stderr, "[kimi_vulkan_batch] n=%d build_us=%.0f set_us=%.0f compute_us=%.0f get_us=%.0f\n",
                count, build_us, set_us, compute_us, get_us);
    }
    ggml_free(ctx);
    return true;
}

bool expert_mul_mat_batch(void * raw, const ggml_vk_expert_matmul * items, int count,
        char * error, size_t error_size) {
    auto * session = static_cast<expert_session *>(raw);
    // The sink already published the finished slot. Joining here would wait for
    // the next layer's fill and then show that slot to this batch.
    if (session) { session->ahead_used = 0; }
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
    if (session->ahead_used) {
        std::fprintf(stderr, "[kimi_vk_ahead] used=%d\n", session->ahead_used);
    }
    return true;
}

constexpr uint64_t k_ahead_cap = 1536ull << 20;

bool expert_ahead(void * raw, const ggml_vk_expert_ahead * items, int count) {
    auto * session = static_cast<expert_session *>(raw);
    if (!session) { return false; }
    if (!items || count <= 0) {
        session->join_ahead();
        return true;
    }
    // The batch is reading the shown slot. Fill the other one during this layer's
    // GEMM and the next layer's attention, so the next join is not the copy.
    if (session->ahead_thread.joinable()) { session->join_ahead(); }
    const int dest = 1 - session->ahead_shown;
    auto & slot = session->ahead_slots[dest];
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session->backend);
    const size_t align = std::max<size_t>(ggml_backend_buft_get_alignment(buft), 256);
    struct planned { const void * key; const void * data; size_t bytes; size_t offset; };
    std::vector<planned> plan;
    plan.reserve(size_t(count));
    size_t cursor = 0;
    for (int i = 0; i < count; ++i) {
        if (!items[i].key || !items[i].data || items[i].bytes == 0) { continue; }
        const size_t off = align_up(cursor, align);
        if (off + items[i].bytes > k_ahead_cap) { break; }
        plan.push_back({ items[i].key, items[i].data, items[i].bytes, off });
        cursor = off + items[i].bytes;
    }
    if (plan.empty()) { return false; }
    // A growing host-visible slot minor-faults the new mapping, and those fills
    // ran at 5–8 GB/s. Both slots are allocated once and touched before the copy.
    auto prefault_slot = [&](int index) {
        auto & held = session->ahead_slots[index];
        if (held.buffer && held.cap >= k_ahead_cap) { return true; }
        if (held.buffer) {
            ggml_backend_buffer_free(held.buffer);
            held.buffer = nullptr;
            held.cap = 0;
            held.at.clear();
            held.bytes.clear();
        }
        try { held.buffer = ggml_backend_buft_alloc_buffer(buft, k_ahead_cap); }
        catch (...) { held.buffer = nullptr; }
        if (!held.buffer) { return false; }
        held.cap = ggml_backend_buffer_get_size(held.buffer);
        void * mapped = ggml_vk_mapped_ptr(held.buffer);
        const auto t0 = std::chrono::steady_clock::now();
        if (mapped && held.cap) {
            auto * bytes = static_cast<volatile char *>(mapped);
            for (size_t off = 0; off < held.cap; off += 4096) { bytes[off] = 0; }
        }
        std::fprintf(stderr, "[kimi_vk_ahead] prefault slot=%d bytes=%zu us=%.0f mapped=%d\n",
                index, held.cap,
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                mapped ? 1 : 0);
        return true;
    };
    if (!prefault_slot(0) || !prefault_slot(1)) {
        if (!slot.buffer || slot.cap < cursor) {
            if (slot.buffer) { ggml_backend_buffer_free(slot.buffer); }
            try { slot.buffer = ggml_backend_buft_alloc_buffer(buft, cursor); }
            catch (...) { slot.buffer = nullptr; }
            slot.cap = slot.buffer ? ggml_backend_buffer_get_size(slot.buffer) : 0;
            if (!slot.buffer) { return false; }
        }
    }
    if (dest == session->ahead_shown) { session->ahead_buffer = slot.buffer; }
    slot.at.clear();
    slot.bytes.clear();
    session->ahead_filling = dest;
    session->ahead_thread = std::thread([session, dest, plan = std::move(plan)]() {
        auto & filled = session->ahead_slots[dest];
        const size_t mem = ggml_tensor_overhead() * plan.size() + 65536;
        ggml_context * ctx = ggml_init({ mem, nullptr, true });
        if (!ctx || !filled.buffer) { if (ctx) { ggml_free(ctx); } return; }
        void * base = ggml_backend_buffer_get_base(filled.buffer);
        const auto t0 = std::chrono::steady_clock::now();
        size_t bytes = 0;
        int queued = 0;
        for (const auto & item : plan) {
            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, int64_t(item.bytes));
            if (!tensor) { break; }
            if (ggml_backend_tensor_alloc(filled.buffer, tensor,
                    static_cast<char *>(base) + item.offset) != GGML_STATUS_SUCCESS) { break; }
            ggml_backend_tensor_set(tensor, item.data, 0, item.bytes);
            filled.at[item.key] = item.offset;
            filled.bytes[item.key] = item.bytes;
            bytes += item.bytes;
            ++queued;
        }
        std::fprintf(stderr, "[kimi_vk_ahead] queued=%d bytes=%zu set_us=%.0f\n", queued, bytes,
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        ggml_free(ctx);
    });
    return true;
}

// Experts with the same padded token count share one mul_mat. That is the
// ordinary shader, one dispatch per bucket, not mul_mat_id. False leaves the
// batch buffer reusable by the per-expert path and does not write output.
bool expert_buckets_compute(expert_session * session, const ggml_tensor * activation,
        int32_t up_type, int32_t down_type, int64_t latent, int64_t hidden,
        size_t up_bytes, size_t down_bytes, const ggml_vk_device_expert * experts, int n_experts,
        std::vector<size_t> & down_off, std::vector<int> & down_cols) {
    if (n_experts < 32) { return false; }
    static const int k_pads[] = { 8, 16, 32, 64, 128, 256, 512 };
    struct bucket { int pad = 0; std::vector<int> which; };
    std::vector<bucket> buckets;
    for (int p : k_pads) {
        bucket b{ p, {} };
        for (int e = 0; e < n_experts; ++e) {
            const int n = experts[e].n_tokens;
            if (!experts[e].up || !experts[e].gate || !experts[e].down || !experts[e].tokens || n < 1) { return false; }
            int pad = 0;
            for (int q : k_pads) { if (n <= q) { pad = q; break; } }
            if (!pad) { return false; }
            if (pad == p) { b.which.push_back(e); }
        }
        if (!b.which.empty()) { buckets.push_back(std::move(b)); }
    }
    if (buckets.empty()) { return false; }
    ggml_context * probe = ggml_init({ ggml_tensor_overhead() * 4 + 4096, nullptr, true });
    if (!probe) { return false; }
    ggml_tensor * up_probe = ggml_new_tensor_3d(probe, ggml_type(up_type), latent, hidden, 1);
    ggml_tensor * down_probe = ggml_new_tensor_3d(probe, ggml_type(down_type), hidden, latent, 1);
    const bool stride_ok = up_probe && down_probe && up_probe->nb[2] == up_bytes && down_probe->nb[2] == down_bytes;
    ggml_free(probe);
    if (!stride_ok) {
        std::fprintf(stderr, "[kimi_vk_bucket] skip stride up=%zu down=%zu\n", up_bytes, down_bytes);
        return false;
    }
    const size_t row_bytes = size_t(latent) * sizeof(float);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session->backend);
    const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), 256);
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * 160 + ggml_graph_overhead_custom(128, false) + 65536, nullptr, true });
    if (!ctx) { return false; }
    struct built {
        int pad = 0;
        std::vector<int> which;
        ggml_tensor * up_w = nullptr;
        ggml_tensor * gate_w = nullptr;
        ggml_tensor * down_w = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * up_out = nullptr;
        ggml_tensor * gate_out = nullptr;
        ggml_tensor * hidden_out = nullptr;
        ggml_tensor * down_out = nullptr;
        size_t up_at = 0, gate_at = 0, down_at = 0, in_at = 0;
        size_t up_out_at = 0, gate_out_at = 0, hidden_at = 0, down_out_at = 0;
    };
    std::vector<built> layers;
    layers.reserve(buckets.size());
    for (const auto & b : buckets) {
        const int count = int(b.which.size());
        built lay;
        lay.pad = b.pad;
        lay.which = b.which;
        lay.up_w = ggml_new_tensor_3d(ctx, ggml_type(up_type), latent, hidden, count);
        lay.gate_w = ggml_new_tensor_3d(ctx, ggml_type(up_type), latent, hidden, count);
        lay.down_w = ggml_new_tensor_3d(ctx, ggml_type(down_type), hidden, latent, count);
        lay.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, latent, b.pad, count);
        if (!lay.up_w || !lay.gate_w || !lay.down_w || !lay.input ||
                lay.up_w->nb[2] != up_bytes || lay.down_w->nb[2] != down_bytes ||
                lay.input->nb[1] != row_bytes) {
            ggml_free(ctx);
            return false;
        }
        lay.up_out = ggml_mul_mat(ctx, lay.up_w, lay.input);
        lay.gate_out = ggml_mul_mat(ctx, lay.gate_w, lay.input);
        lay.hidden_out = lay.up_out && lay.gate_out ? ggml_swiglu_split(ctx, lay.gate_out, lay.up_out) : nullptr;
        lay.down_out = lay.hidden_out ? ggml_mul_mat(ctx, lay.down_w, lay.hidden_out) : nullptr;
        if (!lay.down_out || lay.down_out->nb[1] != row_bytes || lay.down_out->nb[2] != size_t(b.pad) * row_bytes) {
            ggml_free(ctx);
            return false;
        }
        layers.push_back(std::move(lay));
    }
    size_t cursor = 0;
    auto take = [&](ggml_tensor * tensor) -> size_t {
        cursor = align_up(cursor, alignment);
        const size_t at = cursor;
        cursor += ggml_backend_buft_get_alloc_size(buft, tensor);
        return at;
    };
    for (auto & lay : layers) {
        lay.up_at = take(lay.up_w);
        lay.gate_at = take(lay.gate_w);
        lay.down_at = take(lay.down_w);
    }
    const size_t input_at = align_up(cursor, alignment);
    for (auto & lay : layers) {
        lay.in_at = take(lay.input);
    }
    const size_t input_bytes = cursor - input_at;
    for (auto & lay : layers) {
        lay.up_out_at = take(lay.up_out);
        lay.gate_out_at = take(lay.gate_out);
        lay.hidden_at = take(lay.hidden_out);
        lay.down_out_at = take(lay.down_out);
    }
    if (cursor == 0 || cursor > k_batch_cap || input_bytes == 0 || input_bytes % 4 != 0) {
        std::fprintf(stderr, "[kimi_vk_bucket] skip experts=%d bytes=%zu cap=%zu\n",
                n_experts, cursor, size_t(k_batch_cap));
        ggml_free(ctx);
        return false;
    }
    if (!session->batch_buffer || session->batch_staging < cursor || session->batch_buft != buft) {
        if (session->batch_buffer) { ggml_backend_buffer_free(session->batch_buffer); }
        session->batch_buffer = ggml_backend_buft_alloc_buffer(buft, cursor);
        session->batch_buft = buft;
        session->batch_staging = session->batch_buffer ? ggml_backend_buffer_get_size(session->batch_buffer) : 0;
        if (!session->batch_buffer) {
            ggml_free(ctx);
            return false;
        }
    }
    const size_t act_base = size_t(static_cast<char *>(activation->data) - static_cast<char *>(ggml_backend_buffer_get_base(activation->buffer)));
    std::vector<size_t> act_src, act_dst, act_sz;
    std::vector<size_t> w_src, w_dst, w_sz;
    std::vector<const void *> host_src;
    std::vector<size_t> host_dst, host_sz;
    int misses = 0;
    auto place_weight = [&](const void * data, size_t bytes, size_t dst) {
        const auto ahead = session->ahead_at.find(data);
        if (ahead != session->ahead_at.end() && session->ahead_buffer && session->ahead_bytes[data] >= bytes &&
                ahead->second + bytes <= ggml_backend_buffer_get_size(session->ahead_buffer)) {
            w_src.push_back(ahead->second);
            w_dst.push_back(dst);
            w_sz.push_back(bytes);
            return;
        }
        host_src.push_back(data);
        host_dst.push_back(dst);
        host_sz.push_back(bytes);
        ++misses;
    };
    for (const auto & lay : layers) {
        for (size_t i = 0; i < lay.which.size(); ++i) {
            const auto & item = experts[lay.which[i]];
            for (int c = 0; c < item.n_tokens; ++c) {
                const int token = item.tokens[c];
                if (token < 0 || token >= activation->ne[1]) {
                    ggml_free(ctx);
                    return false;
                }
                act_src.push_back(act_base + size_t(token) * row_bytes);
                act_dst.push_back(lay.in_at + (i * size_t(lay.pad) + size_t(c)) * row_bytes);
                act_sz.push_back(row_bytes);
            }
            place_weight(item.up, up_bytes, lay.up_at + i * up_bytes);
            place_weight(item.gate, up_bytes, lay.gate_at + i * up_bytes);
            place_weight(item.down, down_bytes, lay.down_at + i * down_bytes);
        }
    }
    const auto t_pack = std::chrono::steady_clock::now();
    if (!ggml_vk_fill_and_copy(session->batch_buffer, input_at, input_bytes, activation->buffer,
                act_src.data(), act_dst.data(), act_sz.data(), int(act_src.size()))) {
        ggml_free(ctx);
        return false;
    }
    if (!w_src.empty() && !ggml_vk_fill_and_copy(session->batch_buffer, 0, 0, session->ahead_buffer,
                w_src.data(), w_dst.data(), w_sz.data(), int(w_src.size()))) {
        ggml_free(ctx);
        return false;
    }
    if (!host_src.empty() && !ggml_vk_write_ranges(session->batch_buffer, host_src.data(), host_dst.data(), host_sz.data(), int(host_src.size()))) {
        ggml_free(ctx);
        return false;
    }
    const double pack_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_pack).count();
    void * base = ggml_backend_buffer_get_base(session->batch_buffer);
    auto place = [&](ggml_tensor * tensor, size_t at) -> bool {
        return tensor && ggml_backend_tensor_alloc(session->batch_buffer, tensor, static_cast<char *>(base) + at) == GGML_STATUS_SUCCESS;
    };
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    if (!graph) {
        ggml_free(ctx);
        return false;
    }
    for (const auto & lay : layers) {
        if (!place(lay.up_w, lay.up_at) || !place(lay.gate_w, lay.gate_at) || !place(lay.down_w, lay.down_at) ||
                !place(lay.input, lay.in_at) || !place(lay.up_out, lay.up_out_at) || !place(lay.gate_out, lay.gate_out_at) ||
                !place(lay.hidden_out, lay.hidden_at) || !place(lay.down_out, lay.down_out_at)) {
            ggml_free(ctx);
            return false;
        }
        for (size_t i = 0; i < lay.which.size(); ++i) {
            const int e = lay.which[i];
            down_off[size_t(e)] = lay.down_out_at + i * size_t(lay.pad) * row_bytes;
            down_cols[size_t(e)] = experts[e].n_tokens;
        }
        ggml_build_forward_expand(graph, lay.down_out);
    }
    std::fprintf(stderr, "[kimi_vk_bucket] begin experts=%d buckets=%zu misses=%d bytes=%zu\n",
            n_experts, layers.size(), misses, cursor);
    std::fflush(stderr);
    const auto t_compute = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(session->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[kimi_vk_bucket] compute failed\n");
        std::fflush(stderr);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_synchronize(session->backend);
    const double compute_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_compute).count();
    std::fprintf(stderr, "[kimi_vk_bucket] experts=%d buckets=%zu misses=%d pack_us=%.0f compute_us=%.0f\n",
            n_experts, layers.size(), misses, pack_us, compute_us);
    std::fflush(stderr);
    ggml_free(ctx);
    return true;
}

bool expert_device_layer(void * raw, const ggml_tensor * activation, int32_t up_type, int32_t down_type,
        int64_t latent, int64_t hidden, size_t up_bytes, size_t down_bytes,
        const ggml_vk_device_expert * experts, int n_experts, const ggml_vk_device_route * routes, int n_routes,
        int nslots, int ntokens, float * output, char * error, size_t error_size) {
    auto * session = static_cast<expert_session *>(raw);
    if (!session || !activation || !experts || n_experts < 1 || !routes || n_routes < 1 || !output ||
            nslots < 1 || nslots > 16 || ntokens < 1 || latent < 1 || hidden < 1) {
        set_error(error, error_size, "vulkan device layer is missing buffers");
        return false;
    }
    if (!ggml_vk_tensor_on_device(activation) || activation->ne[0] != latent || activation->ne[1] < ntokens) {
        set_error(error, error_size, "vulkan device layer activation is not a device tensor");
        return false;
    }
    const size_t row_bytes = size_t(latent) * sizeof(float);
    const size_t slot_need = size_t(ntokens) * row_bytes;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session->backend);
    if (session->slot_bytes < slot_need) {
        for (auto & buf : session->slot_buf) {
            if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        }
        session->slot_bytes = 0;
        for (int s = 0; s < nslots; ++s) {
            session->slot_buf[s] = ggml_backend_buft_alloc_buffer(buft, slot_need);
            if (!session->slot_buf[s]) {
                set_error(error, error_size, "vulkan device layer slot buffer failed");
                return false;
            }
        }
        session->slot_bytes = slot_need;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), 128);
    struct placed { ggml_tensor * tensor = nullptr; size_t offset = 0; bool ahead = false; const void * host = nullptr; size_t bytes = 0; };
    std::vector<int> down_cols(size_t(n_experts), 0);
    std::vector<size_t> down_off(size_t(n_experts), 0);
    double compute_us = 0;
    if (expert_buckets_compute(session, activation, up_type, down_type, latent, hidden,
                up_bytes, down_bytes, experts, n_experts, down_off, down_cols)) {
        compute_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    } else {
    const int graph_nodes = n_experts * 32 + 256;
    const size_t mem = ggml_tensor_overhead() * size_t(n_experts) * 12 + ggml_graph_overhead_custom(graph_nodes, false) + 65536;
    ggml_context * ctx = ggml_init({ mem, nullptr, true });
    if (!ctx) {
        set_error(error, error_size, "vulkan device layer context failed");
        return false;
    }
    std::vector<placed> sets;
    std::vector<ggml_tensor *> downs(size_t(n_experts), nullptr);
    std::vector<const void *> id_src;
    std::vector<size_t> id_off;
    std::vector<size_t> id_bytes;
    ggml_tensor * act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, activation->ne[0], activation->ne[1]);
    if (!act) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan device layer activation view failed");
        return false;
    }
    // Same bytes as the dense-graph tensor. Not a producer, so the expert graph
    // does not record the layers above it.
    act->buffer = activation->buffer;
    act->data = activation->data;
    size_t cursor = 0;
    auto take = [&](ggml_tensor * tensor) -> bool {
        if (!tensor) { return false; }
        cursor = align_up(cursor, alignment);
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
        if (cursor + bytes > k_batch_cap) { return false; }
        sets.push_back({ tensor, cursor, false, nullptr, 0 });
        cursor += bytes;
        return true;
    };
    for (int e = 0; e < n_experts; ++e) {
        const auto & item = experts[e];
        if (!item.up || !item.gate || !item.down || !item.tokens || item.n_tokens < 1) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan device layer expert is incomplete");
            return false;
        }
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, item.n_tokens);
        ggml_tensor * rows = ids ? ggml_get_rows(ctx, act, ids) : nullptr;
        ggml_tensor * up_w = ggml_new_tensor_2d(ctx, ggml_type(up_type), latent, hidden);
        ggml_tensor * gate_w = ggml_new_tensor_2d(ctx, ggml_type(up_type), latent, hidden);
        ggml_tensor * up_out = up_w && rows ? ggml_mul_mat(ctx, up_w, rows) : nullptr;
        ggml_tensor * gate_out = gate_w && rows ? ggml_mul_mat(ctx, gate_w, rows) : nullptr;
        ggml_tensor * hidden_out = up_out && gate_out ? ggml_swiglu_split(ctx, gate_out, up_out) : nullptr;
        ggml_tensor * down_w = ggml_new_tensor_2d(ctx, ggml_type(down_type), hidden, latent);
        ggml_tensor * down_out = down_w && hidden_out ? ggml_mul_mat(ctx, down_w, hidden_out) : nullptr;
        if (!down_out || ggml_nbytes(up_w) != up_bytes || ggml_nbytes(down_w) != down_bytes) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan device layer graph build failed");
            return false;
        }
        if (!take(ids)) { ggml_free(ctx); set_error(error, error_size, "vulkan device layer staging cap"); return false; }
        id_src.push_back(item.tokens);
        id_off.push_back(sets.back().offset);
        id_bytes.push_back(size_t(item.n_tokens) * sizeof(int32_t));
        auto place_weight = [&](ggml_tensor * weight, const void * data, size_t bytes) -> bool {
            const auto ahead = session->ahead_at.find(data);
            if (ahead != session->ahead_at.end() && session->ahead_buffer &&
                    session->ahead_bytes[data] >= bytes &&
                    ahead->second + bytes <= ggml_backend_buffer_get_size(session->ahead_buffer)) {
                sets.push_back({ weight, ahead->second, true, nullptr, 0 });
                return true;
            }
            if (!take(weight)) { return false; }
            sets.back().host = data;
            sets.back().bytes = bytes;
            return true;
        };
        if (!place_weight(up_w, item.up, up_bytes) || !place_weight(gate_w, item.gate, up_bytes) ||
                !place_weight(down_w, item.down, down_bytes) || !take(rows) || !take(up_out) ||
                !take(gate_out) || !take(hidden_out) || !take(down_out)) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan device layer staging cap");
            return false;
        }
        downs[size_t(e)] = down_out;
        down_cols[size_t(e)] = item.n_tokens;
        down_off[size_t(e)] = sets.back().offset;
    }
    if (!session->batch_buffer || session->batch_staging < cursor || session->batch_buft != buft) {
        if (session->batch_buffer) { ggml_backend_buffer_free(session->batch_buffer); }
        session->batch_buffer = ggml_backend_buft_alloc_buffer(buft, cursor);
        session->batch_buft = buft;
        session->batch_staging = session->batch_buffer ? ggml_backend_buffer_get_size(session->batch_buffer) : 0;
        if (!session->batch_buffer) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan device layer staging allocation failed");
            return false;
        }
    }
    void * base = ggml_backend_buffer_get_base(session->batch_buffer);
    void * ahead_base = session->ahead_buffer ? ggml_backend_buffer_get_base(session->ahead_buffer) : nullptr;
    for (const auto & item : sets) {
        ggml_backend_buffer_t buf = item.ahead ? session->ahead_buffer : session->batch_buffer;
        void * at = item.ahead ? static_cast<char *>(ahead_base) + item.offset : static_cast<char *>(base) + item.offset;
        if (ggml_backend_tensor_alloc(buf, item.tensor, at) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            set_error(error, error_size, "vulkan device layer placement failed");
            return false;
        }
        if (item.host && item.bytes) { ggml_backend_tensor_set(item.tensor, item.host, 0, item.bytes); }
    }
    if (!id_src.empty() && !ggml_vk_write_ranges(session->batch_buffer, id_src.data(), id_off.data(), id_bytes.data(), int(id_src.size()))) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan device layer id upload failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan device layer graph allocation failed");
        return false;
    }
    for (auto * down : downs) { ggml_build_forward_expand(graph, down); }
    if (ggml_backend_graph_compute(session->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        set_error(error, error_size, "vulkan device layer graph compute failed");
        return false;
    }
    ggml_backend_synchronize(session->backend);
    compute_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ggml_free(ctx);
    }

    const auto t1 = std::chrono::steady_clock::now();
    std::vector<size_t> src_off;
    std::vector<int> dst_row;
    std::vector<int> slot_n(size_t(nslots), 0);
    src_off.reserve(size_t(n_routes));
    dst_row.reserve(size_t(n_routes));
    for (int s = 0; s < nslots; ++s) {
        const int before = int(src_off.size());
        for (int r = 0; r < n_routes; ++r) {
            const auto & route = routes[r];
            if (route.slot != s || route.expert < 0 || route.expert >= n_experts || route.token < 0 || route.token >= ntokens ||
                    route.column < 0 || route.column >= down_cols[size_t(route.expert)]) {
                continue;
            }
            src_off.push_back(down_off[size_t(route.expert)] + size_t(route.column) * row_bytes);
            dst_row.push_back(route.token);
        }
        slot_n[size_t(s)] = int(src_off.size()) - before;
    }
    if (!ggml_vk_scatter_slots(session->slot_buf, nslots, slot_need, session->batch_buffer,
                src_off.data(), dst_row.data(), slot_n.data(), row_bytes)) {
        set_error(error, error_size, "vulkan device layer scatter failed");
        return false;
    }
    const auto scatter_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();

    const auto t2 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> slot_w(size_t(nslots), std::vector<float>(size_t(ntokens), 0.0f));
    for (int r = 0; r < n_routes; ++r) {
        if (routes[r].slot < 0 || routes[r].slot >= nslots || routes[r].token < 0 || routes[r].token >= ntokens) { continue; }
        slot_w[size_t(routes[r].slot)][size_t(routes[r].token)] = routes[r].weight;
    }
    const int reduce_nodes = nslots * 8 + 32;
    ggml_context * reduce = ggml_init({ ggml_tensor_overhead() * size_t(nslots) * 8 + ggml_graph_overhead_custom(reduce_nodes, false) + 65536, nullptr, true });
    if (!reduce) {
        set_error(error, error_size, "vulkan device layer reduce context failed");
        return false;
    }
    ggml_tensor * sum = nullptr;
    size_t reduce_cursor = 0;
    std::vector<placed> reduce_sets;
    std::vector<ggml_tensor *> slot_tensors(size_t(nslots), nullptr);
    auto take_reduce = [&](ggml_tensor * tensor, const void * host, size_t bytes) -> bool {
        if (!tensor) { return false; }
        reduce_cursor = align_up(reduce_cursor, alignment);
        const size_t need = ggml_backend_buft_get_alloc_size(buft, tensor);
        if (reduce_cursor + need > k_batch_cap) { return false; }
        reduce_sets.push_back({ tensor, reduce_cursor, false, host, bytes });
        reduce_cursor += need;
        return true;
    };
    for (int s = 0; s < nslots; ++s) {
        ggml_tensor * slot = ggml_new_tensor_2d(reduce, GGML_TYPE_F32, latent, ntokens);
        ggml_tensor * weights = ggml_new_tensor_2d(reduce, GGML_TYPE_F32, 1, ntokens);
        ggml_tensor * repeated = slot && weights ? ggml_repeat(reduce, weights, slot) : nullptr;
        ggml_tensor * scaled = slot && repeated ? ggml_mul(reduce, slot, repeated) : nullptr;
        slot_tensors[size_t(s)] = slot;
        if (!scaled || !take_reduce(weights, slot_w[size_t(s)].data(), slot_w[size_t(s)].size() * sizeof(float)) ||
                !take_reduce(repeated, nullptr, 0) || !take_reduce(scaled, nullptr, 0)) {
            ggml_free(reduce);
            set_error(error, error_size, "vulkan device layer reduce build failed");
            return false;
        }
        sum = sum ? ggml_add(reduce, sum, scaled) : scaled;
        if (sum != scaled && !take_reduce(sum, nullptr, 0)) {
            ggml_free(reduce);
            set_error(error, error_size, "vulkan device layer reduce build failed");
            return false;
        }
    }
    if (!sum || reduce_cursor == 0) {
        ggml_free(reduce);
        set_error(error, error_size, "vulkan device layer reduce is empty");
        return false;
    }
    if (!session->batch_buffer || session->batch_staging < reduce_cursor) {
        if (session->batch_buffer) { ggml_backend_buffer_free(session->batch_buffer); }
        session->batch_buffer = ggml_backend_buft_alloc_buffer(buft, reduce_cursor);
        session->batch_staging = session->batch_buffer ? ggml_backend_buffer_get_size(session->batch_buffer) : 0;
        if (!session->batch_buffer) {
            ggml_free(reduce);
            set_error(error, error_size, "vulkan device layer reduce allocation failed");
            return false;
        }
    }
    void * reduce_base = ggml_backend_buffer_get_base(session->batch_buffer);
    for (int s = 0; s < nslots; ++s) {
        if (ggml_backend_tensor_alloc(session->slot_buf[s], slot_tensors[size_t(s)],
                    ggml_backend_buffer_get_base(session->slot_buf[s])) != GGML_STATUS_SUCCESS) {
            ggml_free(reduce);
            set_error(error, error_size, "vulkan device layer slot placement failed");
            return false;
        }
    }
    for (const auto & item : reduce_sets) {
        if (ggml_backend_tensor_alloc(session->batch_buffer, item.tensor,
                    static_cast<char *>(reduce_base) + item.offset) != GGML_STATUS_SUCCESS) {
            ggml_free(reduce);
            set_error(error, error_size, "vulkan device layer reduce placement failed");
            return false;
        }
        if (item.host && item.bytes) { ggml_backend_tensor_set(item.tensor, item.host, 0, item.bytes); }
    }
    ggml_cgraph * reduce_graph = ggml_new_graph_custom(reduce, reduce_nodes, false);
    if (!reduce_graph) {
        ggml_free(reduce);
        set_error(error, error_size, "vulkan device layer reduce graph failed");
        return false;
    }
    ggml_build_forward_expand(reduce_graph, sum);
    if (ggml_backend_graph_compute(session->backend, reduce_graph) != GGML_STATUS_SUCCESS) {
        ggml_free(reduce);
        set_error(error, error_size, "vulkan device layer reduce compute failed");
        return false;
    }
    ggml_backend_synchronize(session->backend);
    ggml_backend_tensor_get(sum, output, 0, slot_need);
    const auto reduce_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t2).count();
    std::fprintf(stderr, "[kimi_vk_device] experts=%d routes=%d compute_us=%.0f scatter_us=%.0f reduce_us=%.0f\n",
            n_experts, n_routes, compute_us, scatter_us, reduce_us);
    ggml_free(reduce);
    return true;
}

// Decode weight cache. Hot experts stay in device memory so a one-token layer
// does not walk their bytes on the CPU. Misses stay on the CPU walk. Uploads
// copy off the caller's pointers immediately and finish on a side thread, so a
// later token sees the expert without waiting on this one.
struct decode_cache {
    static constexpr int k_layers = 48;
    static constexpr int k_experts = 512;
    static constexpr int k_max_n = 16;
    // A full slab is about 2GB. Binding that as one storage buffer corrupts later
    // dispatches on this driver. One window stays under the 128MB Vulkan minimum
    // (28 * 4308480 = 120.6MB). Sixteen slots left a third of a warm token on the
    // CPU walk; the matmul still pads to 16 because a token uses 10 experts.
    static constexpr int k_window = 28;
    static constexpr size_t k_align = 256;
    static constexpr size_t k_max_up = 1126400;
    static constexpr size_t k_max_down = 1740800;
    static constexpr size_t k_gate_off = 1126400;
    static constexpr size_t k_down_off = 2252800;
    // The vec shader indexes experts in whole quant blocks, so the slot stride
    // has to be a multiple of every block size stored here and of the 256-byte
    // buffer alignment. Q4_K is 144, Q5_K is 176, Q5_1 is 24, Q8_0 is 34.
    // LCM of those is 430848. Packed slot end is 3993600; 10 * 430848 = 4308480.
    // A stride that misses Q5_K (layer 2 up/gate) makes the shader read the
    // next expert at the packed size and the matmul comes back non-finite.
    static constexpr size_t k_stride = 4308480;
    static_assert(k_stride % 144 == 0, "Q4_K block");
    static_assert(k_stride % 176 == 0, "Q5_K block");
    static_assert(k_stride % 24 == 0, "Q5_1 block");
    static_assert(k_stride % 34 == 0, "Q8_0 block");
    static_assert(k_stride % 256 == 0, "alignment");
    static_assert(k_stride >= k_down_off + k_max_down, "slot fits");

    struct slot_rec {
        int32_t key = -1;
        uint16_t freq = 0;
        uint32_t stamp = 0;
        int slab = -1;
        size_t offset = 0;
        bool ready = false;
        bool busy = false;
        // 0 while an acquire is deferred, 1 once the compute queue can read it,
        // -1 if the upload failed. Not atomic-typed: the slot vector is resized.
        int acquire_done = 1;
    };
    struct slab_rec {
        ggml_backend_buffer_t buffer = nullptr;
        int nslots = 0;
    };
    struct layer_home {
        int begin = -1;
        int slab = -1;
        int base = 0;
        // Overflow window. Same 28-slot bind; a second chain only when it has hits.
        int begin2 = -1;
        int slab2 = -1;
        int base2 = 0;
    };
    struct job {
        int slot = -1;
        ggml_backend_buffer_t buffer = nullptr;
        size_t offset = 0;
        // Copied on the worker. The host-cache slot stays until this layer's
        // admit, which is after the worker has been woken.
        const void * up = nullptr;
        const void * gate = nullptr;
        const void * down = nullptr;
        size_t up_bytes = 0;
        size_t down_bytes = 0;
    };
    // Owned copy of a job. The host-cache pointers are only valid until the
    // worker copies them, and that copy stays off the miss walk.
    struct staged {
        int slot = -1;
        ggml_backend_buffer_t buffer = nullptr;
        size_t offset = 0;
        std::vector<uint8_t> up;
        std::vector<uint8_t> gate;
        std::vector<uint8_t> down;
    };
    struct span {
        ggml_backend_buffer_t buffer = nullptr;
        size_t offset = 0;
    };
    struct gpu_piece {
        int index = -1;
        int slab = -1;
        int local = -1;
        span up, gate, down;
        bool imported = false;
    };
    struct import_rec {
        uintptr_t base = 0;
        size_t bytes = 0;
        ggml_backend_buffer_t buffer = nullptr;
    };
    struct built {
        int32_t up_type = -1;
        int32_t down_type = -1;
        int64_t latent = 0;
        int64_t hidden = 0;
        int n = 0;
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_tensor * input = nullptr;
        ggml_backend_buffer_t scratch = nullptr;
        void * mapped = nullptr;
        bool batched = false;
        ggml_backend_buffer_t pack = nullptr;
        size_t pack_up = 0, pack_gate = 0, pack_down = 0;
        ggml_tensor * down_batched = nullptr;
        std::vector<ggml_tensor *> up_w, gate_w, down_w, down_out;
        struct chain {
            ggml_tensor * ids = nullptr;
            ggml_tensor * down = nullptr;
            int n = 0;
            ggml_tensor * up_w = nullptr;
            ggml_tensor * gate_w = nullptr;
            ggml_tensor * down_w = nullptr;
            int span = 0;
        };
        // One mul_mat_id chain per slab window. Weights stay in the cache; ids are slot indexes.
        bool inplace = false;
        std::vector<chain> chains;
        std::vector<int> key;
        uint32_t used = 0;
    };

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    std::vector<slab_rec> slabs;
    std::vector<slot_rec> slots;
    // One full window per layer, inside a single slab. Ten hits then share one
    // matmul. A stripe that crossed a slab boundary scattered them into several
    // chains, and each chain was a new graph.
    std::vector<layer_home> homes;
    std::unordered_map<int32_t, int> at;
    std::vector<uint16_t> seen;
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::deque<job> jobs;
    std::vector<int> pending_ready;
    std::atomic<int> upload_busy{0};
    std::thread worker;
    bool stop = false;
    bool failed = false;
    bool live = false;
    uint32_t stamp = 0;
    int admits_window = 0;
    int evicts_window = 0;
    int window_layers = 0;
    int stat_hits = 0;
    int stat_n = 0;
    double stat_record = 0;
    double stat_wait = 0;
    double stat_copy = 0;

    ggml_vk_decode_batch * held = nullptr;
    std::vector<int> hit_of;
    std::vector<gpu_piece> plan;
    std::vector<import_rec> imports;
    bool import_off = true;
    bool batched_broken = false;
    bool inplace_broken = false;
    bool plan_imports = false;
    bool launched_inplace = false;
    std::vector<int> launched_real;
    std::vector<built> inplace_pool;
    int active_inplace = -1;
    uint32_t pool_clock = 0;
    int stat_reuse = 0;
    int stat_build = 0;
    int stat_import = 0;
    int stat_cold = 0;
    int stat_full = 0;
    int stat_reclaim = 0;
    int stat_pending = 0;
    int stat_chains = 0;
    int stat_multi = 0;
    std::atomic<int> stat_upload_n{0};
    std::atomic<int64_t> stat_upload_us{0};
    std::atomic<int> stat_upload_late{0};
    int graph_index = -1;
    bool launched = false;
    std::atomic<int> vulkan_busy{0};
    std::vector<built> graphs;
    std::vector<float> raw;

    ~decode_cache() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
            if (live_cache.load(std::memory_order_acquire) == this) {
                live_cache.store(nullptr, std::memory_order_release);
            }
        }
        cv.notify_all();
        idle_cv.notify_all();
        if (worker.joinable()) { worker.join(); }
        for (auto & g : graphs) {
            if (g.scratch) { ggml_backend_buffer_free(g.scratch); }
            if (g.pack) { ggml_backend_buffer_free(g.pack); }
            if (g.ctx) { ggml_free(g.ctx); }
        }
        for (auto & g : inplace_pool) { drop_built(g); }
        inplace_pool.clear();
        active_inplace = -1;
        for (auto & slab : slabs) {
            if (slab.buffer) { ggml_backend_buffer_free(slab.buffer); }
        }
        for (auto & rec : imports) {
            if (rec.buffer) { ggml_backend_buffer_free(rec.buffer); }
        }
    }

    void fail_staged(std::vector<staged> & held) {
        std::lock_guard<std::mutex> lock(mu);
        for (auto & item : held) {
            if (item.slot < 0 || item.slot >= int(slots.size())) { continue; }
            auto & slot = slots[size_t(item.slot)];
            slot.busy = false;
            slot.ready = false;
            if (slot.key >= 0) {
                at.erase(slot.key);
                slot.key = -1;
            }
            __atomic_store_n(&slot.acquire_done, -1, __ATOMIC_RELEASE);
        }
        held.clear();
    }

    void flush_staged(std::vector<staged> & held) {
        if (held.empty()) { return; }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<size_t> order(held.size());
        for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return held[a].buffer < held[b].buffer;
        });
        size_t begin = 0;
        while (begin < order.size()) {
            ggml_backend_buffer_t buffer = held[order[begin]].buffer;
            size_t end = begin + 1;
            while (end < order.size() && held[order[end]].buffer == buffer) { ++end; }
            std::vector<size_t> offs;
            std::vector<const void *> srcs;
            std::vector<size_t> sizes;
            std::vector<int *> flags;
            std::vector<size_t> good;
            std::vector<size_t> bad;
            for (size_t k = begin; k < end; ++k) {
                auto & item = held[order[k]];
                if (!buffer || item.slot < 0 || item.slot >= int(slots.size()) ||
                        item.up.empty() || item.gate.empty() || item.down.empty()) {
                    bad.push_back(k);
                    continue;
                }
                offs.push_back(item.offset);
                srcs.push_back(item.up.data());
                sizes.push_back(item.up.size());
                offs.push_back(item.offset + k_gate_off);
                srcs.push_back(item.gate.data());
                sizes.push_back(item.gate.size());
                offs.push_back(item.offset + k_down_off);
                srcs.push_back(item.down.data());
                sizes.push_back(item.down.size());
                flags.push_back(&slots[size_t(item.slot)].acquire_done);
                good.push_back(k);
            }
            const int visible = !offs.empty() ?
                ggml_vk_upload_host_ranges(buffer, offs.data(), srcs.data(), sizes.data(),
                        int(offs.size()), flags.data(), int(flags.size())) : 0;
            auto drop_slot = [&](size_t k) {
                auto & item = held[order[k]];
                if (item.slot < 0 || item.slot >= int(slots.size())) { return; }
                auto & slot = slots[size_t(item.slot)];
                slot.busy = false;
                slot.ready = false;
                if (slot.key >= 0) {
                    at.erase(slot.key);
                    slot.key = -1;
                }
                __atomic_store_n(&slot.acquire_done, -1, __ATOMIC_RELEASE);
            };
            std::lock_guard<std::mutex> lock(mu);
            for (size_t k : bad) { drop_slot(k); }
            for (size_t k : good) {
                if (!visible) { drop_slot(k); continue; }
                auto & item = held[order[k]];
                auto & slot = slots[size_t(item.slot)];
                slot.busy = false;
                slot.ready = true;
            }
            begin = end;
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
        stat_upload_us.fetch_add(us, std::memory_order_relaxed);
        stat_upload_n.fetch_add(int(held.size()), std::memory_order_relaxed);
        held.clear();
    }

    bool walk_busy() const {
        return decode_walk_active.load(std::memory_order_acquire) ||
            vulkan_busy.load(std::memory_order_acquire);
    }

    void fail_slot_locked(int index) {
        if (index < 0 || index >= int(slots.size())) { return; }
        auto & slot = slots[size_t(index)];
        slot.busy = false;
        slot.ready = false;
        if (slot.key >= 0) {
            at.erase(slot.key);
            slot.key = -1;
        }
        __atomic_store_n(&slot.acquire_done, -1, __ATOMIC_RELEASE);
    }

    void mark_ready_locked(int index) {
        if (index < 0 || index >= int(slots.size())) { return; }
        auto & slot = slots[size_t(index)];
        __atomic_store_n(&slot.acquire_done, 1, __ATOMIC_RELEASE);
        slot.busy = false;
        slot.ready = slot.key >= 0;
    }

    // One expert per GPU fence wait. The copy and the DMA both read host DRAM,
    // so they stay inside that wait. A yield-spin here was a seventh runnable
    // thread on a 6-core CPU for the whole miss walk.
    void worker_main() {
        for (;;) {
            job item;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] {
                    return stop || (upload_window_flag.load(std::memory_order_acquire) && !jobs.empty());
                });
                if (stop) { return; }
                if (!upload_window_flag.load(std::memory_order_acquire) || jobs.empty()) { continue; }
                item = std::move(jobs.front());
                jobs.pop_front();
                upload_busy.store(1, std::memory_order_release);
            }
            int rc = 0;
            int64_t us = 0;
            const bool pointers = item.buffer && item.up && item.gate && item.down && item.up_bytes && item.down_bytes;
            if (!pointers) {
                rc = 0;
            } else if (upload_window_flag.load(std::memory_order_acquire) &&
                    !decode_walk_active.load(std::memory_order_acquire)) {
                const auto t0 = std::chrono::steady_clock::now();
                rc = ggml_vk_upload_expert_fast(item.buffer,
                        item.offset, item.up, item.up_bytes,
                        item.offset + k_gate_off, item.gate, item.up_bytes,
                        item.offset + k_down_off, item.down, item.down_bytes);
                us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
            } else {
                rc = -1;
            }
            {
                std::lock_guard<std::mutex> lock(mu);
                if (rc == -1) {
                    jobs.push_front(std::move(item));
                } else if (rc == 1) {
                    mark_ready_locked(item.slot);
                    stat_upload_n.fetch_add(1, std::memory_order_relaxed);
                    stat_upload_us.fetch_add(us, std::memory_order_relaxed);
                } else if (rc == 2) {
                    pending_ready.push_back(item.slot);
                    stat_upload_n.fetch_add(1, std::memory_order_relaxed);
                    stat_upload_us.fetch_add(us, std::memory_order_relaxed);
                } else {
                    fail_slot_locked(item.slot);
                }
                if (rc > 0 && !upload_window_flag.load(std::memory_order_acquire)) {
                    stat_upload_late.fetch_add(1, std::memory_order_relaxed);
                }
                upload_busy.store(0, std::memory_order_release);
            }
            idle_cv.notify_all();
            if (rc == -1) {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] {
                    return stop || !upload_window_flag.load(std::memory_order_acquire);
                });
            } else if (rc > 0) {
                // One expert per window. A second submit at the end of the wait
                // runs its DMA into the miss walk.
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] {
                    return stop || !upload_window_flag.load(std::memory_order_acquire);
                });
            }
            if (stop) { return; }
        }
    }

    bool ensure(expert_session * session) {
        if (live) { return true; }
        if (failed) { return false; }
        const char * off = std::getenv("INFERSSD_DECODE_GPU");
        if (off && off[0] == '0' && off[1] == '\0') { failed = true; return false; }
        const char * inplace = std::getenv("INFERSSD_DECODE_INPLACE");
        if (inplace && inplace[0] == '0' && inplace[1] == '\0') { inplace_broken = true; }
        // Importing every miss copies about a gigabyte of tiny regions and
        // makes a token take seconds. Misses stay on the CPU walk.
        import_off = true;
        const char * imp = std::getenv("INFERSSD_DECODE_IMPORT");
        if (imp && imp[0] == '1' && imp[1] == '\0') { import_off = false; }
        session->join_ahead();
        // These slots are device-local. Some drivers still report them in the
        // budget query after the free, which left the decode cache 3 GB small.
        // A driver that updates immediately must not be credited twice.
        backend = session->backend;
        size_t before_free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &before_free, &total);
        size_t returned = 0;
        for (auto & ahead : session->ahead_slots) {
            if (ahead.buffer) {
                returned += ggml_backend_buffer_get_size(ahead.buffer);
                ggml_backend_buffer_free(ahead.buffer);
                ahead.buffer = nullptr;
                ahead.cap = 0;
            }
            ahead.at.clear();
            ahead.bytes.clear();
        }
        session->ahead_buffer = nullptr;
        session->ahead_at.clear();
        session->ahead_bytes.clear();
        buft = ggml_backend_get_default_buffer_type(backend);
        size_t free_bytes = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_bytes, &total);
        if (free_bytes < before_free + returned) { free_bytes = before_free + returned; }
        constexpr size_t reserve = 256ull << 20;
        size_t budget = free_bytes > reserve ? free_bytes - reserve : 0;
        if (budget > (10ull << 30)) { budget = 10ull << 30; }
        const int want = int(budget / k_stride);
        if (want < 64 || !buft) {
            failed = true;
            std::fprintf(stderr, "[kimi_vk_decode] no cache free=%zu budget=%zu\n", free_bytes, budget);
            return false;
        }
        size_t left = size_t(want) * k_stride;
        constexpr size_t slab_cap = 2ull << 30;
        while (left >= k_stride) {
            size_t chunk = std::min(left, slab_cap - (slab_cap % k_stride));
            chunk = (chunk / k_stride) * k_stride;
            if (chunk < k_stride) { break; }
            ggml_backend_buffer_t buf = nullptr;
            try { buf = ggml_backend_buft_alloc_buffer(buft, chunk); }
            catch (...) { buf = nullptr; }
            if (!buf && chunk > k_stride * 128) {
                chunk = k_stride * 128;
                try { buf = ggml_backend_buft_alloc_buffer(buft, chunk); }
                catch (...) { buf = nullptr; }
            }
            if (!buf) { break; }
            slab_rec slab;
            slab.buffer = buf;
            slab.nslots = int(ggml_backend_buffer_get_size(buf) / k_stride);
            const int base = int(slots.size());
            slots.resize(size_t(base + slab.nslots));
            for (int i = 0; i < slab.nslots; ++i) {
                slots[size_t(base + i)].slab = int(slabs.size());
                slots[size_t(base + i)].offset = size_t(i) * k_stride;
            }
            slabs.push_back(slab);
            left -= size_t(slab.nslots) * k_stride;
        }
        if (slots.size() < 64) {
            failed = true;
            std::fprintf(stderr, "[kimi_vk_decode] allocation failed free=%zu slabs=%zu\n", free_bytes, slabs.size());
            return false;
        }
        seen.assign(size_t(k_layers * k_experts), 0);
        graphs.reserve(8);
        homes.assign(k_layers, {});
        struct window_rec { int begin; int slab; int base; };
        std::vector<window_rec> windows;
        int global = 0;
        for (int s = 0; s < int(slabs.size()); ++s) {
            const int n = slabs[size_t(s)].nslots;
            for (int base = 0; base + k_window <= n; base += k_window) {
                windows.push_back({ global + base, s, base });
            }
            global += n;
        }
        const int primary = std::min(k_layers, int(windows.size()));
        for (int i = 0; i < primary; ++i) {
            homes[size_t(i)] = { windows[size_t(i)].begin, windows[size_t(i)].slab, windows[size_t(i)].base, -1, -1, 0 };
        }
        int extra = 0;
        for (int i = primary; i < int(windows.size()) && extra < k_layers; ++i, ++extra) {
            auto & home = homes[size_t(extra)];
            home.begin2 = windows[size_t(i)].begin;
            home.slab2 = windows[size_t(i)].slab;
            home.base2 = windows[size_t(i)].base;
        }
        int assigned = primary;
        live_cache.store(this, std::memory_order_release);
        worker = std::thread([this] { worker_main(); });
        live = true;
        std::fprintf(stderr, "[kimi_vk_decode] cache slots=%zu stride=%zu free_was=%zu windows=%d extra=%d\n",
                slots.size(), k_stride, free_bytes, assigned, extra);
        return true;
    }

    // Each layer owns a run of windows. Ten hot experts then share one matmul
    // instead of a chain per scattered slot, which is what made recording the token.
    int layer_stripe(int layer) const {
        const int stripe = (int(slots.size()) / k_layers / k_window) * k_window;
        return stripe >= k_window ? stripe : 0;
    }

    int free_in_range_locked(int begin, int end) const {
        if (begin < 0) { begin = 0; }
        if (end > int(slots.size())) { end = int(slots.size()); }
        for (int i = begin; i < end; ++i) {
            if (slots[size_t(i)].key < 0 && !slots[size_t(i)].busy) { return i; }
        }
        return -1;
    }

    int same_window_free_locked(int32_t key) const {
        const int layer = key / k_experts;
        if (layer < 0 || layer >= k_layers) { return -1; }
        const int stripe = layer_stripe(layer);
        const int origin = stripe ? layer * stripe : 0;
        const int limit = stripe ? std::min(origin + stripe, int(slots.size())) : int(slots.size());
        int anchor = -1;
        for (const auto & ent : at) {
            if (ent.first / k_experts != layer || ent.second < origin || ent.second >= limit) { continue; }
            anchor = ent.second;
            break;
        }
        if (anchor >= 0) {
            const auto & home = slots[size_t(anchor)];
            if (home.slab >= 0 && home.slab < int(slabs.size())) {
                const int local = int(home.offset / k_stride);
                const int base = (local / k_window) * k_window;
                const int end = std::min(base + k_window, slabs[size_t(home.slab)].nslots);
                for (int i = origin; i < limit; ++i) {
                    const auto & slot = slots[size_t(i)];
                    if (slot.slab != home.slab || slot.key >= 0 || slot.busy) { continue; }
                    const int loc = int(slot.offset / k_stride);
                    if (loc >= base && loc < end) { return i; }
                }
            }
        }
        return free_in_range_locked(origin, limit);
    }

    int free_slot_locked() const {
        for (int i = 0; i < int(slots.size()); ++i) {
            if (slots[size_t(i)].key < 0 && !slots[size_t(i)].busy) { return i; }
        }
        return -1;
    }

    int victim_in_range_locked(int begin, int end) const {
        int victim = -1;
        uint16_t best_freq = 65535;
        uint32_t best_stamp = UINT32_MAX;
        if (begin < 0) { begin = 0; }
        if (end > int(slots.size())) { end = int(slots.size()); }
        for (int i = begin; i < end; ++i) {
            const auto & slot = slots[size_t(i)];
            if (slot.key < 0 || slot.busy || slot.stamp == stamp) { continue; }
            if (slot.freq < best_freq || (slot.freq == best_freq && slot.stamp < best_stamp)) {
                best_freq = slot.freq;
                best_stamp = slot.stamp;
                victim = i;
            }
        }
        return victim;
    }

    // stamp advances once per layer, 48 times a token. Six quiet tokens still
    // left the window full of experts used inside that window, so twice-seen
    // newcomers stayed misses. Two quiet tokens is not LRU: anything used in
    // the last two tokens keeps its slot.
    static constexpr uint32_t k_quiet = 48u * 2u;

    int stale_in_range_locked(int begin, int end) const {
        int victim = -1;
        uint32_t best_gap = 0;
        if (begin < 0) { begin = 0; }
        if (end > int(slots.size())) { end = int(slots.size()); }
        for (int i = begin; i < end; ++i) {
            const auto & slot = slots[size_t(i)];
            if (slot.key < 0 || slot.busy) { continue; }
            const uint32_t gap = stamp - slot.stamp;
            if (gap < k_quiet) { continue; }
            if (victim < 0 || gap > best_gap ||
                    (gap == best_gap && slot.freq < slots[size_t(victim)].freq)) {
                best_gap = gap;
                victim = i;
            }
        }
        return victim;
    }

    int victim_locked() const {
        return victim_in_range_locked(0, int(slots.size()));
    }

    void enqueue_locked(int index, const void * up, const void * gate, const void * down, size_t up_bytes, size_t down_bytes) {
        auto & slot = slots[size_t(index)];
        job item;
        item.slot = index;
        item.buffer = slabs[size_t(slot.slab)].buffer;
        item.offset = slot.offset;
        item.up = up;
        item.gate = gate;
        item.down = down;
        item.up_bytes = up_bytes;
        item.down_bytes = down_bytes;
        slot.busy = true;
        slot.ready = false;
        __atomic_store_n(&slot.acquire_done, 0, __ATOMIC_RELEASE);
        jobs.push_back(std::move(item));
        ++admits_window;
        cv.notify_one();
    }

    bool admit_locked(int32_t key, uint16_t freq, const void * up, const void * gate, const void * down,
            size_t up_bytes, size_t down_bytes) {
        // A one-shot expert never earns a slot. The first-come fill left every
        // warm token with a miss even when the cache was full.
        if (freq < 2) { return false; }
        // One expert per GPU fence wait. A deeper queue only occupies slots
        // that are not ready yet. Cap 48 admitted more and the mean did not move.
        if (jobs.size() >= 24) { return false; }
        const int layer = key / k_experts;
        if (layer < 0 || layer >= int(homes.size()) || homes[size_t(layer)].begin < 0) { return false; }
        const auto & home = homes[size_t(layer)];
        const int begin = home.begin;
        const int end = std::min(begin + k_window, int(slots.size()));
        const int begin2 = home.begin2;
        const int end2 = begin2 >= 0 ? std::min(begin2 + k_window, int(slots.size())) : -1;
        auto take_free = [&](int from, int to) {
            for (int i = from; i < to; ++i) {
                if (slots[size_t(i)].key < 0 && !slots[size_t(i)].busy) { return i; }
            }
            return -1;
        };
        auto take_victim = [&](int from, int to) {
            const int found = victim_in_range_locked(from, to);
            if (found >= 0 && freq > slots[size_t(found)].freq) { return found; }
            // A prompt-hot expert keeps a huge lifetime count. Reclaim it only
            // after it has been quiet for many tokens, so a working-set expert
            // is not evicted to make room for a newcomer.
            return stale_in_range_locked(from, to);
        };
        // Fill the primary window first so a layer whose hot set fits in 28 slots
        // stays one chain. Overflow uses the second window before evicting a hot one.
        int index = take_free(begin, end);
        bool took_victim = false;
        if (index < 0 && end2 > begin2) { index = take_free(begin2, end2); }
        // Evictions copy during the GPU fence wait and finish before the miss
        // walk. Raising this from 32 to 48 admitted more and the token did not
        // get faster: the extra evictions were not the misses on the mean.
        const bool can_evict = evicts_window < 32;
        if (can_evict && index < 0 && end2 > begin2) {
            index = take_victim(begin2, end2);
            took_victim = index >= 0;
        }
        if (can_evict && index < 0) {
            index = take_victim(begin, end);
            if (index < 0) { return false; }
            took_victim = true;
        }
        if (index < 0) { return false; }
        if (took_victim) { ++evicts_window; }
        if (took_victim) {
            const auto & old = slots[size_t(index)];
            if (stamp - old.stamp >= k_quiet && freq <= old.freq) { ++stat_reclaim; }
            at.erase(old.key);
        }
        auto & slot = slots[size_t(index)];
        slot.key = key;
        slot.freq = freq;
        slot.stamp = stamp;
        at[key] = index;
        enqueue_locked(index, up, gate, down, up_bytes, down_bytes);
        return true;
    }

    // A type whose block does not divide the slot stride cannot be indexed by
    // the vec shader. Leave those experts on the CPU walk instead of reading
    // the wrong bytes.
    static bool stride_fits(int32_t type) {
        const size_t block = ggml_type_size(static_cast<ggml_type>(type));
        return block != 0 && k_stride % block == 0;
    }

    static void bind_weight(ggml_tensor * tensor, ggml_backend_buffer_t buffer, size_t offset) {
        tensor->buffer = buffer;
        tensor->data = static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + offset;
    }

    static void drop_built(built & g) {
        if (g.scratch) { ggml_backend_buffer_free(g.scratch); g.scratch = nullptr; }
        if (g.pack) { ggml_backend_buffer_free(g.pack); g.pack = nullptr; }
        if (g.ctx) { ggml_free(g.ctx); g.ctx = nullptr; }
        g.gf = nullptr;
        g.input = nullptr;
        g.down_batched = nullptr;
        g.chains.clear();
        g.up_w.clear();
        g.gate_w.clear();
        g.down_w.clear();
        g.down_out.clear();
        g.inplace = false;
        g.key.clear();
    }

    struct slab_group {
        int slab = -1;
        int base = 0;
        std::vector<int32_t> local;
        std::vector<int> hit;
    };

    // Shape only: window count and how many hits sit in each. The slab and base
    // are a rebind, not a new graph. Rebuilding every layer was the 25 ms.
    std::vector<int> shape_key(const ggml_vk_decode_batch * batch, const std::vector<slab_group> & groups) {
        std::vector<int> key;
        key.reserve(5 + groups.size() * 2);
        key.push_back(batch->up_type);
        key.push_back(batch->down_type);
        key.push_back(int(batch->latent));
        key.push_back(int(batch->hidden));
        key.push_back(int(groups.size()));
        for (const auto & group : groups) {
            if (group.slab < 0 || group.slab >= int(slabs.size())) {
                key.push_back(-1);
                key.push_back(int(group.local.size()));
                continue;
            }
            const int span = std::min(k_window, slabs[size_t(group.slab)].nslots - group.base);
            key.push_back(span);
            key.push_back(int(group.local.size()));
        }
        return key;
    }

    bool rebind_inplace(built & g, const std::vector<slab_group> & groups) {
        if (g.chains.size() != groups.size()) { return false; }
        for (size_t c = 0; c < groups.size(); ++c) {
            const auto & group = groups[c];
            auto & chain = g.chains[c];
            if (!chain.up_w || !chain.gate_w || !chain.down_w || !chain.ids) { return false; }
            if (group.slab < 0 || group.slab >= int(slabs.size())) { return false; }
            const auto & slab = slabs[size_t(group.slab)];
            const int span = std::min(k_window, slab.nslots - group.base);
            if (!slab.buffer || span != chain.span || int(group.local.size()) != chain.n) { return false; }
            for (int32_t slot : group.local) {
                if (slot < 0 || slot >= span) { return false; }
            }
            const size_t origin = size_t(group.base) * k_stride;
            char * base = static_cast<char *>(ggml_backend_buffer_get_base(slab.buffer));
            auto poke = [&](ggml_tensor * tensor, size_t role) {
                tensor->buffer = slab.buffer;
                tensor->data = base + origin + role;
            };
            poke(chain.up_w, 0);
            poke(chain.gate_w, k_gate_off);
            poke(chain.down_w, k_down_off);
        }
        return true;
    }

    // Hits in one slab share a buffer and a slot stride, so one mul_mat_id reads
    // them where they sit. No gather copy.
    built * ensure_inplace(const ggml_vk_decode_batch * batch, const std::vector<slab_group> & groups) {
        if (groups.empty() || !buft || !backend || !batch) { return nullptr; }
        const std::vector<int> key = shape_key(batch, groups);
        for (int part : key) {
            if (part < 0) { return nullptr; }
        }
        int hits = 0;
        for (const auto & group : groups) { hits += int(group.local.size()); }
        for (size_t i = 0; i < inplace_pool.size(); ++i) {
            auto & held = inplace_pool[i];
            if (!held.inplace || held.key != key) { continue; }
            if (!rebind_inplace(held, groups)) { return nullptr; }
            held.used = ++pool_clock;
            active_inplace = int(i);
            ++stat_reuse;
            return &inplace_pool[i];
        }
        if (inplace_pool.size() >= 48) {
            size_t oldest = 0;
            for (size_t i = 1; i < inplace_pool.size(); ++i) {
                if (inplace_pool[i].used < inplace_pool[oldest].used) { oldest = i; }
            }
            drop_built(inplace_pool[oldest]);
            inplace_pool.erase(inplace_pool.begin() + std::ptrdiff_t(oldest));
            if (active_inplace == int(oldest)) { active_inplace = -1; }
            else if (active_inplace > int(oldest)) { --active_inplace; }
        }
        built g;
        g.inplace = true;
        g.key = key;
        g.up_type = batch->up_type;
        g.down_type = batch->down_type;
        g.latent = batch->latent;
        g.hidden = batch->hidden;
        g.n = hits;
        auto fail = [&](const char * why) -> built * {
            static int logged = 0;
            if (!logged++) { std::fprintf(stderr, "[kimi_vk_decode] inplace unavailable: %s\n", why); std::fflush(stderr); }
            drop_built(g);
            return nullptr;
        };
        const int nchains = int(groups.size());
        const int nodes = nchains * 8 + 8;
        const size_t mem = ggml_tensor_overhead() * size_t(nchains) * 12 + ggml_graph_overhead_custom(nodes, false) + 65536;
        g.ctx = ggml_init({ mem, nullptr, true });
        if (!g.ctx) { return fail("context"); }
        g.input = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, batch->latent, 1, 1);
        if (!g.input) { return fail("input"); }
        const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), k_align);
        struct scratch_item { ggml_tensor * tensor = nullptr; size_t off = 0; };
        std::vector<scratch_item> scratch;
        size_t cursor = 0;
        auto take = [&](ggml_tensor * tensor) -> bool {
            if (!tensor) { return false; }
            cursor = align_up(cursor, alignment);
            scratch.push_back({ tensor, cursor });
            cursor += ggml_backend_buft_get_alloc_size(buft, tensor);
            return true;
        };
        if (!take(g.input)) { return fail("input place"); }
        auto view_weight = [&](ggml_tensor * tensor, ggml_backend_buffer_t buffer, size_t role) -> bool {
            if (!tensor || !buffer) { return false; }
            tensor->nb[2] = k_stride;
            tensor->nb[3] = k_stride * size_t(tensor->ne[2]);
            const size_t need = ggml_nbytes(tensor);
            if (role + need > ggml_backend_buffer_get_size(buffer)) {
                std::fprintf(stderr, "[kimi_vk_decode] inplace view role=%zu need=%zu slab=%zu\n",
                        role, need, ggml_backend_buffer_get_size(buffer));
                return false;
            }
            void * base = ggml_backend_buffer_get_base(buffer);
            return ggml_backend_tensor_alloc(buffer, tensor, static_cast<char *>(base) + role) == GGML_STATUS_SUCCESS;
        };
        g.chains.resize(size_t(nchains));
        for (int c = 0; c < nchains; ++c) {
            const auto & group = groups[size_t(c)];
            if (group.slab < 0 || group.slab >= int(slabs.size()) || group.local.empty()) { return fail("slab"); }
            const auto & slab = slabs[size_t(group.slab)];
            const int n = int(group.local.size());
            const int span = std::min(k_window, slab.nslots - group.base);
            if (span < 1 || n > k_max_n || group.base < 0) { return fail("count"); }
            for (int32_t slot : group.local) {
                if (slot < 0 || slot >= span) { return fail("slot"); }
            }
            const size_t origin = size_t(group.base) * k_stride;
            ggml_tensor * up_w = ggml_new_tensor_3d(g.ctx, ggml_type(batch->up_type), batch->latent, batch->hidden, span);
            ggml_tensor * gate_w = ggml_new_tensor_3d(g.ctx, ggml_type(batch->up_type), batch->latent, batch->hidden, span);
            ggml_tensor * down_w = ggml_new_tensor_3d(g.ctx, ggml_type(batch->down_type), batch->hidden, batch->latent, span);
            ggml_tensor * ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n, 1);
            if (!view_weight(up_w, slab.buffer, origin) || !view_weight(gate_w, slab.buffer, origin + k_gate_off) ||
                    !view_weight(down_w, slab.buffer, origin + k_down_off) || !ids) {
                return fail("weight view");
            }
            ggml_tensor * up_out = ggml_mul_mat_id(g.ctx, up_w, g.input, ids);
            ggml_tensor * gate_out = ggml_mul_mat_id(g.ctx, gate_w, g.input, ids);
            ggml_tensor * hidden = up_out && gate_out ? ggml_swiglu_split(g.ctx, gate_out, up_out) : nullptr;
            ggml_tensor * down_out = hidden ? ggml_mul_mat_id(g.ctx, down_w, hidden, ids) : nullptr;
            if (!down_out || !take(ids) || !take(up_out) || !take(gate_out) || !take(hidden) || !take(down_out)) {
                return fail("chain");
            }
            if (!ggml_backend_supports_op(backend, up_out) || !ggml_backend_supports_op(backend, gate_out) ||
                    !ggml_backend_supports_op(backend, hidden) || !ggml_backend_supports_op(backend, down_out)) {
                std::fprintf(stderr, "[kimi_vk_decode] inplace op unsupported nbytes=%zu\n", ggml_nbytes(up_w));
                return fail("op");
            }
            g.chains[size_t(c)] = { ids, down_out, n, up_w, gate_w, down_w, span };
        }
        g.scratch = nullptr;
        try { g.scratch = ggml_backend_buft_alloc_buffer(buft, cursor); }
        catch (...) { g.scratch = nullptr; }
        if (!g.scratch) { return fail("scratch"); }
        void * base = ggml_backend_buffer_get_base(g.scratch);
        for (const auto & item : scratch) {
            if (ggml_backend_tensor_alloc(g.scratch, item.tensor, static_cast<char *>(base) + item.off) != GGML_STATUS_SUCCESS) {
                return fail("scratch place");
            }
        }
        g.gf = ggml_new_graph_custom(g.ctx, nodes, false);
        if (!g.gf) { return fail("graph"); }
        for (const auto & chain : g.chains) { ggml_build_forward_expand(g.gf, chain.down); }
        g.used = ++pool_clock;
        ++stat_build;
        inplace_pool.push_back(std::move(g));
        active_inplace = int(inplace_pool.size()) - 1;
        if (stat_build <= 8) {
            std::fprintf(stderr, "[kimi_vk_decode] inplace graph slabs=%d hits=%d pool=%zu stride=%zu\n",
                nchains, hits, inplace_pool.size(), k_stride);
            std::fflush(stderr);
        }
        return &inplace_pool[size_t(active_inplace)];
    }

    // Host-cache bytes are an anonymous arena. Import a window so a miss can be
    // copied by the GPU instead of walked on the CPU.
    bool cover(const void * ptr, size_t bytes, ggml_backend_buffer_t & buffer, size_t & offset) {
        if (!ptr || !bytes || import_off || !backend) { return false; }
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        for (const auto & rec : imports) {
            if (p >= rec.base && p + bytes <= rec.base + rec.bytes && rec.buffer) {
                buffer = rec.buffer;
                offset = size_t(p - rec.base);
                return true;
            }
        }
        static const uintptr_t aligns[] = { 4096, 65536 };
        for (uintptr_t align : aligns) {
            const uintptr_t base = p & ~(align - 1);
            const size_t need = size_t((p + bytes) - base);
            const size_t size = (need + align - 1) & ~(align - 1);
            if (size == 0 || size > (16ull << 20)) { continue; }
            ggml_backend_buffer_t buf = nullptr;
            try {
                buf = ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(backend),
                        reinterpret_cast<void *>(base), size, size);
            } catch (...) {
                buf = nullptr;
            }
            if (!buf) { continue; }
            imports.push_back({ base, size, buf });
            buffer = buf;
            offset = size_t(p - base);
            static int logged = 0;
            if (!logged++) { std::fprintf(stderr, "[kimi_vk_decode] host import ok align=%zu\n", size_t(align)); }
            return true;
        }
        static int failed_log = 0;
        if (!failed_log++) { std::fprintf(stderr, "[kimi_vk_decode] host import unavailable\n"); }
        return false;
    }

    bool cover_expert(const void * up, const void * gate, const void * down, size_t up_bytes, size_t down_bytes, gpu_piece & piece) {
        const char * a = static_cast<const char *>(up);
        const char * b = static_cast<const char *>(gate);
        const char * c = static_cast<const char *>(down);
        if (!a || !b || !c) { return false; }
        const char * lo = std::min(a, std::min(b, c));
        const char * hi = std::max(a + up_bytes, std::max(b + up_bytes, c + down_bytes));
        if (hi < lo || size_t(hi - lo) > (8ull << 20)) {
            return cover(up, up_bytes, piece.up.buffer, piece.up.offset) &&
                    cover(gate, up_bytes, piece.gate.buffer, piece.gate.offset) &&
                    cover(down, down_bytes, piece.down.buffer, piece.down.offset);
        }
        ggml_backend_buffer_t buf = nullptr;
        size_t base_off = 0;
        if (!cover(lo, size_t(hi - lo), buf, base_off)) { return false; }
        piece.up.buffer = piece.gate.buffer = piece.down.buffer = buf;
        piece.up.offset = base_off + size_t(a - lo);
        piece.gate.offset = base_off + size_t(b - lo);
        piece.down.offset = base_off + size_t(c - lo);
        return true;
    }

    // One mul_mat chain for the layer. mul_mat_id crashed on the first hit, and
    // a per-expert graph re-records every weight.
    built * graph_for_batched(const ggml_vk_decode_batch * batch, int n) {
        for (auto & g : graphs) {
            if (g.batched && g.up_type == batch->up_type && g.down_type == batch->down_type &&
                    g.latent == batch->latent && g.hidden == batch->hidden && g.n == n) {
                return &g;
            }
        }
        if (n < 1 || n > k_max_n || !buft) { return nullptr; }
        built g;
        g.batched = true;
        g.up_type = batch->up_type;
        g.down_type = batch->down_type;
        g.latent = batch->latent;
        g.hidden = batch->hidden;
        g.n = n;
        const size_t up_bytes = batch->up_bytes;
        const size_t down_bytes = batch->down_bytes;
        const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), k_align);
        auto fail = [&]() -> built * {
            if (g.scratch) { ggml_backend_buffer_free(g.scratch); }
            if (g.pack) { ggml_backend_buffer_free(g.pack); }
            if (g.ctx) { ggml_free(g.ctx); }
            return nullptr;
        };
        const int nodes = 16;
        const size_t mem = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(nodes, false) + 65536;
        g.ctx = ggml_init({ mem, nullptr, true });
        if (!g.ctx) { return nullptr; }
        ggml_tensor * up_w = ggml_new_tensor_3d(g.ctx, ggml_type(batch->up_type), batch->latent, batch->hidden, n);
        ggml_tensor * gate_w = ggml_new_tensor_3d(g.ctx, ggml_type(batch->up_type), batch->latent, batch->hidden, n);
        ggml_tensor * down_w = ggml_new_tensor_3d(g.ctx, ggml_type(batch->down_type), batch->hidden, batch->latent, n);
        g.input = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, batch->latent, 1, n);
        ggml_tensor * up_out = up_w && g.input ? ggml_mul_mat(g.ctx, up_w, g.input) : nullptr;
        ggml_tensor * gate_out = gate_w && g.input ? ggml_mul_mat(g.ctx, gate_w, g.input) : nullptr;
        ggml_tensor * hidden = up_out && gate_out ? ggml_swiglu_split(g.ctx, gate_out, up_out) : nullptr;
        ggml_tensor * down_out = down_w && hidden ? ggml_mul_mat(g.ctx, down_w, hidden) : nullptr;
        const size_t row = size_t(batch->latent) * sizeof(float);
        if (!down_out || !up_w || !down_w || !g.input || up_w->nb[2] != up_bytes || down_w->nb[2] != down_bytes ||
                g.input->nb[1] != row || down_out->nb[2] != row) {
            static int logged = 0;
            if (!logged++) {
                std::fprintf(stderr, "[kimi_vk_decode] matmul stride up=%zu nb2=%zu down=%zu nb2=%zu out=%zu\n",
                        up_bytes, up_w ? up_w->nb[2] : 0, down_bytes, down_w ? down_w->nb[2] : 0,
                        down_out ? down_out->nb[2] : 0);
            }
            return fail();
        }
        g.pack_up = 0;
        g.pack_gate = align_up(size_t(n) * up_bytes, alignment);
        g.pack_down = align_up(g.pack_gate + size_t(n) * up_bytes, alignment);
        const size_t pack_bytes = align_up(g.pack_down + size_t(n) * down_bytes, alignment);
        g.pack = ggml_backend_buft_alloc_buffer(buft, pack_bytes);
        if (!g.pack) { return fail(); }
        size_t cursor = 0;
        auto take = [&](ggml_tensor * tensor) -> size_t {
            cursor = align_up(cursor, alignment);
            const size_t at = cursor;
            cursor += ggml_backend_buft_get_alloc_size(buft, tensor);
            return at;
        };
        const size_t in_at = take(g.input);
        const size_t up_out_at = take(up_out);
        const size_t gate_out_at = take(gate_out);
        const size_t hidden_at = take(hidden);
        const size_t down_out_at = take(down_out);
        g.scratch = ggml_backend_buft_alloc_buffer(buft, cursor);
        if (!g.scratch) { return fail(); }
        void * pack_base = ggml_backend_buffer_get_base(g.pack);
        void * scratch_base = ggml_backend_buffer_get_base(g.scratch);
        auto place = [&](ggml_backend_buffer_t buffer, void * base, ggml_tensor * tensor, size_t offset) -> bool {
            return tensor && ggml_backend_tensor_alloc(buffer, tensor, static_cast<char *>(base) + offset) == GGML_STATUS_SUCCESS;
        };
        if (!place(g.pack, pack_base, up_w, g.pack_up) || !place(g.pack, pack_base, gate_w, g.pack_gate) ||
                !place(g.pack, pack_base, down_w, g.pack_down) || !place(g.scratch, scratch_base, g.input, in_at) ||
                !place(g.scratch, scratch_base, up_out, up_out_at) || !place(g.scratch, scratch_base, gate_out, gate_out_at) ||
                !place(g.scratch, scratch_base, hidden, hidden_at) || !place(g.scratch, scratch_base, down_out, down_out_at)) {
            return fail();
        }
        g.gf = ggml_new_graph_custom(g.ctx, nodes, false);
        if (!g.gf) { return fail(); }
        ggml_build_forward_expand(g.gf, down_out);
        if (!ggml_backend_supports_op(backend, up_out) || !ggml_backend_supports_op(backend, down_out) ||
                !ggml_backend_supports_op(backend, hidden)) {
            static int logged = 0;
            if (!logged++) { std::fprintf(stderr, "[kimi_vk_decode] matmul graph unsupported\n"); }
            return fail();
        }
        g.down_batched = down_out;
        graphs.push_back(std::move(g));
        std::fprintf(stderr, "[kimi_vk_decode] matmul graph n=%d\n", n);
        std::fflush(stderr);
        return &graphs.back();
    }

    built * graph_for(const ggml_vk_decode_batch * batch, int n) {
        for (auto & g : graphs) {
            if (!g.batched && g.up_type == batch->up_type && g.down_type == batch->down_type && g.latent == batch->latent &&
                    g.hidden == batch->hidden && g.n == n) {
                return &g;
            }
        }
        built g;
        g.up_type = batch->up_type;
        g.down_type = batch->down_type;
        g.latent = batch->latent;
        g.hidden = batch->hidden;
        g.n = n;
        const int nodes = n * 8 + 32;
        const size_t mem = ggml_tensor_overhead() * size_t(n) * 12 + ggml_graph_overhead_custom(nodes, false) + 65536;
        g.ctx = ggml_init({ mem, nullptr, true });
        if (!g.ctx) { return nullptr; }
        g.input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, batch->latent, 1);
        const size_t alignment = std::max<size_t>(ggml_backend_buft_get_alignment(buft), k_align);
        size_t cursor = 0;
        std::vector<std::pair<ggml_tensor *, size_t>> placed;
        auto take = [&](ggml_tensor * tensor) -> bool {
            if (!tensor) { return false; }
            cursor = align_up(cursor, alignment);
            placed.push_back({ tensor, cursor });
            cursor += ggml_backend_buft_get_alloc_size(buft, tensor);
            return true;
        };
        if (!take(g.input)) { ggml_free(g.ctx); return nullptr; }
        g.up_w.resize(size_t(n));
        g.gate_w.resize(size_t(n));
        g.down_w.resize(size_t(n));
        g.down_out.resize(size_t(n));
        for (int i = 0; i < n; ++i) {
            ggml_tensor * up_w = ggml_new_tensor_2d(g.ctx, ggml_type(batch->up_type), batch->latent, batch->hidden);
            ggml_tensor * gate_w = ggml_new_tensor_2d(g.ctx, ggml_type(batch->up_type), batch->latent, batch->hidden);
            ggml_tensor * up_out = up_w && g.input ? ggml_mul_mat(g.ctx, up_w, g.input) : nullptr;
            ggml_tensor * gate_out = gate_w && g.input ? ggml_mul_mat(g.ctx, gate_w, g.input) : nullptr;
            ggml_tensor * hidden = up_out && gate_out ? ggml_swiglu_split(g.ctx, gate_out, up_out) : nullptr;
            ggml_tensor * down_w = ggml_new_tensor_2d(g.ctx, ggml_type(batch->down_type), batch->hidden, batch->latent);
            ggml_tensor * down_out = down_w && hidden ? ggml_mul_mat(g.ctx, down_w, hidden) : nullptr;
            if (!down_out || ggml_nbytes(up_w) != batch->up_bytes || ggml_nbytes(down_w) != batch->down_bytes) {
                ggml_free(g.ctx);
                return nullptr;
            }
            if (!take(up_out) || !take(gate_out) || !take(hidden) || !take(down_out)) {
                ggml_free(g.ctx);
                return nullptr;
            }
            g.up_w[size_t(i)] = up_w;
            g.gate_w[size_t(i)] = gate_w;
            g.down_w[size_t(i)] = down_w;
            g.down_out[size_t(i)] = down_out;
        }
        g.scratch = ggml_backend_buft_alloc_buffer(buft, cursor);
        g.mapped = nullptr;
        if (!g.scratch) { ggml_free(g.ctx); return nullptr; }
        void * base = ggml_backend_buffer_get_base(g.scratch);
        for (const auto & item : placed) {
            if (ggml_backend_tensor_alloc(g.scratch, item.first, static_cast<char *>(base) + item.second) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(g.scratch);
                ggml_free(g.ctx);
                return nullptr;
            }
        }
        void * weight_base = ggml_backend_buffer_get_base(slabs[0].buffer);
        for (int i = 0; i < n; ++i) {
            if (ggml_backend_tensor_alloc(slabs[0].buffer, g.up_w[size_t(i)], weight_base) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(slabs[0].buffer, g.gate_w[size_t(i)], weight_base) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(slabs[0].buffer, g.down_w[size_t(i)], weight_base) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(g.scratch);
                ggml_free(g.ctx);
                return nullptr;
            }
        }
        g.gf = ggml_new_graph_custom(g.ctx, nodes, false);
        if (!g.gf) {
            ggml_backend_buffer_free(g.scratch);
            ggml_free(g.ctx);
            return nullptr;
        }
        for (int i = 0; i < n; ++i) { ggml_build_forward_expand(g.gf, g.down_out[size_t(i)]); }
        graphs.push_back(std::move(g));
        return &graphs.back();
    }

    void note(int hits, int n, double record, double wait, double copy) {
        stat_hits += hits;
        stat_n += n;
        stat_record += record;
        stat_wait += wait;
        stat_copy += copy;
        if (++window_layers < k_layers) { return; }
        const int up_n = stat_upload_n.exchange(0, std::memory_order_relaxed);
        const int64_t up_us = stat_upload_us.exchange(0, std::memory_order_relaxed);
        const int up_late = stat_upload_late.exchange(0, std::memory_order_relaxed);
        std::fprintf(stderr, "[kimi_vk_decode] hits=%d/%d import=%d record_us=%.0f wait_us=%.0f copy_us=%.0f slots=%zu/%zu reuse=%d build=%d pool=%zu cold=%d full=%d reclaim=%d pending=%d chains=%d multi=%d upload=%d upload_us=%.0f late=%d\n",
                stat_hits, stat_n, stat_import, stat_record, stat_wait, stat_copy, at.size(), slots.size(),
                stat_reuse, stat_build, inplace_pool.size(), stat_cold, stat_full, stat_reclaim, stat_pending, stat_chains, stat_multi,
                up_n, double(up_us), up_late);
        stat_hits = stat_n = stat_import = stat_reuse = stat_build = 0;
        stat_cold = stat_full = stat_reclaim = stat_pending = stat_chains = stat_multi = 0;
        stat_record = stat_wait = stat_copy = 0;
        window_layers = 0;
        admits_window = 0;
        evicts_window = 0;
    }
};

expert_session::~expert_session() {
    delete decode;
    decode = nullptr;
    join_ahead();
    drop_graph();
    if (buffer) { ggml_backend_buffer_free(buffer); }
    if (batch_buffer) { ggml_backend_buffer_free(batch_buffer); }
    for (auto & slot : ahead_slots) {
        if (slot.buffer) { ggml_backend_buffer_free(slot.buffer); }
    }
    for (auto & buf : slot_buf) {
        if (buf) { ggml_backend_buffer_free(buf); }
    }
}

bool decode_prepare(void * raw, ggml_vk_decode_batch * batch) {
    auto * session = static_cast<expert_session *>(raw);
    if (!session || !batch || !batch->experts || !batch->ran || batch->n_experts < 1 || batch->n_experts > 1024 ||
            !batch->input || batch->latent < 1 || batch->hidden < 1) {
        return false;
    }
    if (!expert_supports(batch->up_type) || !expert_supports(batch->down_type) ||
            batch->up_bytes == 0 || batch->up_bytes > decode_cache::k_max_up ||
            batch->down_bytes == 0 || batch->down_bytes > decode_cache::k_max_down) {
        return false;
    }
    if (!session->decode) { session->decode = new (std::nothrow) decode_cache; }
    if (!session->decode || !session->decode->ensure(session)) { return false; }
    auto * cache = session->decode;
    if (cache->launched) { return false; }
    // Once a token, while the compute queue is idle. Uploads from the previous
    // token become hits before this layer's plan. Doing it every layer put a
    // barrier in front of the dense sync.
    static int prepare_i = 0;
    if ((prepare_i++ % decode_cache::k_layers) == 0) {
        const int acquired = ggml_vk_decode_upload_acquire();
        std::lock_guard<std::mutex> lock(cache->mu);
        for (int index : cache->pending_ready) {
            if (index < 0 || index >= int(cache->slots.size())) { continue; }
            auto & slot = cache->slots[size_t(index)];
            if (!slot.busy) { continue; }
            __atomic_store_n(&slot.acquire_done, acquired ? 1 : -1, __ATOMIC_RELEASE);
        }
        cache->pending_ready.clear();
    }
    {
        std::lock_guard<std::mutex> lock(cache->mu);
        for (auto & slot : cache->slots) {
            if (!slot.busy) { continue; }
            const int state = __atomic_load_n(&slot.acquire_done, __ATOMIC_ACQUIRE);
            if (state == 0) { continue; }
            slot.busy = false;
            if (state > 0 && slot.key >= 0) {
                slot.ready = true;
            } else if (slot.key >= 0) {
                cache->at.erase(slot.key);
                slot.key = -1;
                slot.ready = false;
            }
        }
    }
    std::fill(batch->ran, batch->ran + batch->n_experts, 0);
    cache->held = batch;
    cache->hit_of.clear();
    cache->plan.clear();
    cache->plan_imports = false;
    const int n = std::min(batch->n_experts, decode_cache::k_max_n);
    struct miss { int index; const void * up; const void * gate; const void * down; };
    std::vector<miss> misses;
    // Set before the admit notify. A worker that wakes inside this lock must
    // not memcpy until decode_wait clears the flag.
    decode_walk_active.store(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(cache->mu);
        ++cache->stamp;
        for (int i = 0; i < n; ++i) {
            const auto & expert = batch->experts[i];
            if (expert.layer < 0 || expert.layer >= decode_cache::k_layers || expert.id < 0 ||
                    expert.id >= decode_cache::k_experts || !expert.up || !expert.gate || !expert.down || !expert.output) {
                continue;
            }
            const int32_t key = expert.layer * decode_cache::k_experts + expert.id;
            auto & freq = cache->seen[size_t(key)];
            if (freq < 65535) { ++freq; }
            const auto found = cache->at.find(key);
            if (found != cache->at.end()) {
                auto & slot = cache->slots[size_t(found->second)];
                slot.freq = freq;
                slot.stamp = cache->stamp;
                if (!slot.ready || slot.busy) { ++cache->stat_pending; }
                if (slot.ready && !slot.busy && !cache->inplace_broken &&
                        decode_cache::stride_fits(batch->up_type) &&
                        decode_cache::stride_fits(batch->down_type)) {
                    const int local = int(slot.offset / decode_cache::k_stride);
                    if (slot.slab < 0 || slot.slab >= int(cache->slabs.size()) || local < 0 ||
                            local >= cache->slabs[size_t(slot.slab)].nslots) {
                        continue;
                    }
                    decode_cache::gpu_piece piece;
                    piece.index = i;
                    piece.slab = slot.slab;
                    piece.local = local;
                    piece.up = { cache->slabs[size_t(slot.slab)].buffer, slot.offset };
                    piece.gate = { piece.up.buffer, slot.offset + decode_cache::k_gate_off };
                    piece.down = { piece.up.buffer, slot.offset + decode_cache::k_down_off };
                    cache->plan.push_back(piece);
                    batch->ran[i] = 1;
                    cache->hit_of.push_back(i);
                }
                continue;
            }
            misses.push_back({ i, expert.up, expert.gate, expert.down });
            if (freq < 2) { ++cache->stat_cold; }
            else if (!cache->admit_locked(key, freq, expert.up, expert.gate, expert.down, batch->up_bytes, batch->down_bytes)) {
                ++cache->stat_full;
            }
        }
    }
    // Misses stay on the CPU walk. Importing them copies about a gigabyte of
    // tiny regions and makes a token take seconds.
    (void) misses;
    return true;
}

bool decode_launch(void * raw) {
    auto * session = static_cast<expert_session *>(raw);
    auto * cache = session ? session->decode : nullptr;
    if (!cache || !cache->held || cache->plan.empty()) { return true; }
    if (cache->launched) { return true; }
    if (cache->inplace_broken) { return false; }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<decode_cache::slab_group> groups;
    for (const auto & piece : cache->plan) {
        if (piece.slab < 0 || piece.local < 0) { continue; }
        const int base = (piece.local / decode_cache::k_window) * decode_cache::k_window;
        decode_cache::slab_group * group = nullptr;
        for (auto & existing : groups) {
            if (existing.slab == piece.slab && existing.base == base) { group = &existing; break; }
        }
        if (!group) {
            groups.push_back({});
            group = &groups.back();
            group->slab = piece.slab;
            group->base = base;
        }
        group->local.push_back(int32_t(piece.local - base));
        group->hit.push_back(piece.index);
    }
    if (groups.empty()) { return false; }
    std::sort(groups.begin(), groups.end(), [](const decode_cache::slab_group & a, const decode_cache::slab_group & b) {
        return a.slab < b.slab || (a.slab == b.slab && a.base < b.base);
    });
    cache->stat_chains += int(groups.size());
    if (groups.size() > 1) { ++cache->stat_multi; }
    cache->hit_of.clear();
    cache->launched_real.clear();
    // One chain, padded to a fixed count, is one graph shape. Hit-count changes
    // were rebuilding ~35 graphs a token, and that record was the expert phase.
    // Padding every chain out to the 28-slot window made recording slower.
    // Two windows pad to 4/8/16, not to the window, so the overflow chain stays
    // one reused shape instead of a new graph per hit count.
    const bool bucket_chain = groups.size() == 1 || groups.size() == 2;
    for (auto & group : groups) {
        const int real = int(group.local.size());
        cache->hit_of.insert(cache->hit_of.end(), group.hit.begin(), group.hit.end());
        cache->launched_real.push_back(real);
        if (!bucket_chain || real < 1 || group.slab < 0 || group.slab >= int(cache->slabs.size())) { continue; }
        const int span = std::min(decode_cache::k_window, cache->slabs[size_t(group.slab)].nslots - group.base);
        int bucket = real <= 4 ? 4 : real <= 8 ? 8 : 16;
        if (bucket > span) { bucket = span; }
        if (bucket <= real) { continue; }
        const int32_t pad = group.local.front();
        group.local.reserve(size_t(bucket));
        while (int(group.local.size()) < bucket) { group.local.push_back(pad); }
    }
    cache->vulkan_busy.store(1, std::memory_order_release);
    auto * graph = cache->ensure_inplace(cache->held, groups);
    if (!graph) {
        cache->vulkan_busy.store(0, std::memory_order_release);
        cache->inplace_broken = true;
        return false;
    }
    bool ids_ok = true;
    for (size_t c = 0; c < groups.size() && c < graph->chains.size(); ++c) {
        const auto & group = groups[c];
        auto & chain = graph->chains[c];
        if (!chain.ids || chain.n != int(group.local.size())) { ids_ok = false; break; }
        ggml_backend_tensor_set(chain.ids, group.local.data(), 0, group.local.size() * sizeof(int32_t));
    }
    if (ids_ok && graph->input) {
        ggml_backend_tensor_set(graph->input, cache->held->input, 0, size_t(cache->held->latent) * sizeof(float));
    } else {
        ids_ok = false;
    }
    std::vector<const ggml_tensor *> suffix_tensors;
    std::vector<size_t> suffix_sizes;
    if (ids_ok && graph->inplace && !graph->chains.empty() && cache->held->latent > 0) {
        const size_t row = size_t(cache->held->latent);
        for (size_t c = 0; c < graph->chains.size(); ++c) {
            const auto & chain = graph->chains[c];
            const int real = c < cache->launched_real.size() && cache->launched_real[c] > 0 ?
                cache->launched_real[c] : chain.n;
            if (!chain.down || chain.n < 1 || real < 1 || real > chain.n) {
                suffix_tensors.clear();
                suffix_sizes.clear();
                break;
            }
            suffix_tensors.push_back(chain.down);
            suffix_sizes.push_back(size_t(real) * row * sizeof(float));
        }
    }
    // The output copy is recorded at the end of this command buffer. decode_wait
    // memcpys after the sync instead of submitting a second fence.
    const enum ggml_status computed = ids_ok ? ggml_vk_graph_compute_suffix(session->backend, graph->gf,
            suffix_tensors.empty() ? nullptr : suffix_tensors.data(),
            suffix_sizes.empty() ? nullptr : suffix_sizes.data(),
            int(suffix_tensors.size())) : GGML_STATUS_FAILED;
    cache->vulkan_busy.store(0, std::memory_order_release);
    if (computed != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[kimi_vk_decode] inplace compute failed\n");
        cache->inplace_broken = true;
        return false;
    }
    cache->launched = true;
    cache->launched_inplace = true;
    cache->stat_record += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool decode_wait(void * raw, ggml_vk_decode_batch * batch) {
    decode_walk_active.store(0, std::memory_order_release);
    auto * session = static_cast<expert_session *>(raw);
    auto * cache = session ? session->decode : nullptr;
    if (!cache) { return true; }
    const int hits = cache->launched ? int(cache->hit_of.size()) : 0;
    const int n = batch ? batch->n_experts : 0;
    double wait_us = 0, copy_us = 0;
    bool ok = true;
    if (cache->launched) {
        cache->launched = false;
        const auto t0 = std::chrono::steady_clock::now();
        cache->vulkan_busy.store(1, std::memory_order_release);
        ggml_backend_synchronize(session->backend);
        wait_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        const int count = int(cache->hit_of.size());
        const bool pooled = cache->launched_inplace && cache->active_inplace >= 0 &&
            cache->active_inplace < int(cache->inplace_pool.size());
        const bool have_graph = pooled ||
            (cache->graph_index >= 0 && cache->graph_index < int(cache->graphs.size()));
        if (!batch || !have_graph || count < 1 || batch->latent < 1) {
            ok = false;
        } else {
            auto & graph = pooled ? cache->inplace_pool[size_t(cache->active_inplace)] : cache->graphs[size_t(cache->graph_index)];
            cache->raw.resize(size_t(count) * size_t(batch->latent));
            const auto t1 = std::chrono::steady_clock::now();
            if (graph.inplace && !graph.chains.empty()) {
                std::vector<const ggml_tensor *> tensors;
                std::vector<void *> dsts;
                std::vector<size_t> sizes;
                size_t at = 0;
                const size_t row = size_t(batch->latent);
                for (size_t c = 0; c < graph.chains.size(); ++c) {
                    const auto & chain = graph.chains[c];
                    const int real = c < cache->launched_real.size() && cache->launched_real[c] > 0 ?
                        cache->launched_real[c] : chain.n;
                    if (!chain.down || chain.n < 1 || real < 1 || real > chain.n) { ok = false; break; }
                    tensors.push_back(chain.down);
                    dsts.push_back(cache->raw.data() + at);
                    sizes.push_back(size_t(real) * row * sizeof(float));
                    at += size_t(real) * row;
                }
                ok = ok && at == size_t(count) * row && (
                    ggml_vk_take_suffix_reads(session->backend, dsts.data(), sizes.data(), int(tensors.size())) ||
                    ggml_vk_copy_tensors_fenced(tensors.data(), dsts.data(), sizes.data(), int(tensors.size())));
            } else if (graph.batched && graph.down_batched) {
                const ggml_tensor * one = graph.down_batched;
                void * dst = cache->raw.data();
                const size_t bytes = size_t(count) * size_t(batch->latent) * sizeof(float);
                ok = ggml_vk_copy_tensors_fenced(&one, &dst, &bytes, 1);
            } else {
                std::vector<const ggml_tensor *> tensors(size_t(count), nullptr);
                std::vector<void *> dsts(size_t(count), nullptr);
                std::vector<size_t> sizes(size_t(count), size_t(batch->latent) * sizeof(float));
                for (int j = 0; j < count; ++j) {
                    tensors[size_t(j)] = graph.down_out[size_t(j)];
                    dsts[size_t(j)] = cache->raw.data() + size_t(j) * size_t(batch->latent);
                }
                ok = ggml_vk_copy_tensors_fenced(tensors.data(), dsts.data(), sizes.data(), count);
            }
            copy_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
            if (ok) {
                for (int j = 0; j < count && ok; ++j) {
                    const float * src = cache->raw.data() + size_t(j) * size_t(batch->latent);
                    for (int64_t row = 0; row < batch->latent; ++row) {
                        if (!std::isfinite(src[row])) {
                            const int index = size_t(j) < cache->hit_of.size() ? cache->hit_of[size_t(j)] : -1;
                            const int id = index >= 0 && index < batch->n_experts ? batch->experts[index].id : -1;
                            const int layer = index >= 0 && index < batch->n_experts ? batch->experts[index].layer : -1;
                            std::fprintf(stderr, "[kimi_vk_decode] non-finite output j=%d layer=%d id=%d row=%lld value=%g\n",
                                j, layer, id, static_cast<long long>(row), static_cast<double>(src[row]));
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if (ok) {
                for (int j = 0; j < count; ++j) {
                    const int index = cache->hit_of[size_t(j)];
                    if (index < 0 || index >= batch->n_experts || !batch->experts[index].output) { ok = false; break; }
                    float * dst = batch->experts[index].output;
                    const float weight = batch->experts[index].weight;
                    const float * src = cache->raw.data() + size_t(j) * size_t(batch->latent);
                    for (int64_t row = 0; row < batch->latent; ++row) {
                        volatile float scaled = src[row] * weight;
                        dst[row] = scaled;
                    }
                }
            }
        }
        cache->vulkan_busy.store(0, std::memory_order_release);
    }
    if (cache->live) { cache->note(hits, n, 0, wait_us, copy_us); }
    cache->held = nullptr;
    cache->hit_of.clear();
    cache->plan.clear();
    cache->plan_imports = false;
    cache->launched_inplace = false;
    return ok;
}

}  // namespace

extern "C" void ggml_vk_decode_upload_window(int open) {
    decode_cache * cache = live_cache.load(std::memory_order_acquire);
    if (cache) {
        std::lock_guard<std::mutex> lock(cache->mu);
        upload_window_flag.store(open ? 1 : 0, std::memory_order_release);
        if (open) {
            cache->cv.notify_one();
        } else {
            cache->cv.notify_all();
            cache->idle_cv.notify_all();
        }
    } else {
        upload_window_flag.store(open ? 1 : 0, std::memory_order_release);
    }
    if (!open && cache) {
        std::unique_lock<std::mutex> lock(cache->mu);
        cache->idle_cv.wait(lock, [&] {
            return cache->upload_busy.load(std::memory_order_acquire) == 0;
        });
    }
}

extern "C" int ggml_vk_decode_upload_window_open(void) {
    return upload_window_flag.load(std::memory_order_acquire);
}

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
        reinterpret_cast<bool (*)(void *, const ggml_vk_expert_ahead *, int)>(expert_ahead),
        reinterpret_cast<bool (*)(void *, const ggml_tensor *, int32_t, int32_t, int64_t, int64_t, size_t, size_t, const ggml_vk_device_expert *, int, const ggml_vk_device_route *, int, int, int, float *, char *, size_t)>(expert_device_layer),
        reinterpret_cast<bool (*)(void *, ggml_vk_decode_batch *)>(decode_prepare),
        reinterpret_cast<bool (*)(void *)>(decode_launch),
        reinterpret_cast<bool (*)(void *, ggml_vk_decode_batch *)>(decode_wait),
    };
    return &api;
}
