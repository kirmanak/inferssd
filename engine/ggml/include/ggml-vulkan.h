#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

// One fence for many device-to-host copies. Returns false if the tensors do not
// share a Vulkan device buffer.
GGML_BACKEND_API bool ggml_vk_copy_tensors_fenced(const struct ggml_tensor ** tensors, void ** dsts, const size_t * sizes, int n);

// One graphics-queue copy into an exclusive device buffer, then one acquire on
// the compute queue. The caller runs the shader only after this returns.
GGML_BACKEND_API bool ggml_vk_upload_acquired(ggml_backend_buffer_t dst, size_t dst_offset, const void * src, size_t bytes);

struct ggml_vk_copy_region { size_t src_offset; size_t dst_offset; size_t bytes; };
// Same acquire, many regions, source already a Vulkan buffer. One fence.
GGML_BACKEND_API bool ggml_vk_copy_regions_acquired(ggml_backend_buffer_t dst, ggml_backend_buffer_t src,
        const struct ggml_vk_copy_region * regions, int n);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
