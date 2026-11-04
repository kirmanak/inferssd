// SPDX-License-Identifier: Apache-2.0
#pragma once

// Vulkan prompt-expert matmul. The CPU sink decides which experts fit in the
// device budget and calls mul_mat for those weights. This is not the CUDA
// provider table: no residency, MXFP4, or deferred upload.
#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_vk_expert_api_v1 {
    uint32_t abi;  // 1
    uint32_t size; // sizeof(ggml_vk_expert_api_v1)
    // True when this build's existing quantized mul_mat pipeline covers the type.
    bool (*supports)(int32_t type);
    void * (*session_create)(ggml_backend_t backend, char * error, size_t error_size);
    void (*session_destroy)(void * session);
    // Resident staging buffer high-water, reused across experts. Not a sum over layers.
    uint64_t (*staging_bytes)(void * session);
    // weight is one expert, ggml layout ne0=K ne1=N. input and output are f32,
    // column-major [K, n_tokens] and [N, n_tokens]. Copies before return.
    bool (*mul_mat)(void * session, int32_t type, int64_t k, int64_t n, int n_tokens,
            const void * weight, size_t weight_bytes, const float * input, float * output,
            char * error, size_t error_size);
    // Many independent matmuls, one graph submission. Same layout as mul_mat.
    // Null on an older build; a false return leaves outputs untouched.
    bool (*mul_mat_batch)(void * session, const struct ggml_vk_expert_matmul * items, int count,
            char * error, size_t error_size);
    // Copy these host weights with the existing per-tensor set, on a background
    // thread. Returns before the copies finish. The next mul_mat_batch joins and
    // skips the upload for a matching weight pointer. Null if this build has no
    // ahead path. count 0 only joins an earlier call.
    bool (*ahead)(void * session, const struct ggml_vk_expert_ahead * items, int count);
    // Activations already on this device. Gathers rows, runs up/gate/SwiGLU/down,
    // scale-adds into the token output, and writes only that reduced buffer.
    // False leaves output untouched. Null on an older build.
    bool (*device_layer)(void * session, const struct ggml_tensor * activation,
            int32_t up_type, int32_t down_type, int64_t latent, int64_t hidden,
            size_t up_bytes, size_t down_bytes,
            const struct ggml_vk_device_expert * experts, int n_experts,
            const struct ggml_vk_device_route * routes, int n_routes,
            int nslots, int ntokens, float * output, char * error, size_t error_size);
};

struct ggml_vk_device_expert {
    const void * up;
    const void * gate;
    const void * down;
    const int32_t * tokens;
    int n_tokens;
};

struct ggml_vk_device_route {
    int32_t expert;
    int32_t column;
    int32_t token;
    int32_t slot;
    float weight;
};

struct ggml_vk_expert_ahead {
    const void * key;
    const void * data;
    size_t bytes;
};

// One expert in a mul_mat_batch. output is caller storage, written only on success.
struct ggml_vk_expert_matmul {
    int32_t type;
    int64_t k;
    int64_t n;
    int n_tokens;
    const void * weight;
    size_t weight_bytes;
    const float * input;
    float * output;
};

typedef const struct ggml_vk_expert_api_v1 * (*ggml_vk_expert_api_getter)(void);

const struct ggml_vk_expert_api_v1 * ggml_backend_vk_expert_api_v1(void);

#ifdef __cplusplus
}
#endif
