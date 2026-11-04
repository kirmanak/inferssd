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
GGML_BACKEND_API bool ggml_vk_tensor_on_device(const struct ggml_tensor * tensor);
// Zero dst, then copy rows from src. dst_rows[i] is the destination row. One fence.
GGML_BACKEND_API bool ggml_vk_scatter_rows(ggml_backend_buffer_t dst, size_t dst_bytes,
        ggml_backend_buffer_t src, const size_t * src_offsets, const int * dst_rows,
        int n, size_t row_bytes);
// Same as scatter_rows for every slot, one submit. slot_n[s] is the route count
// for dsts[s]; the route arrays are packed in slot order. A slot with no routes
// is still zeroed.
GGML_BACKEND_API bool ggml_vk_scatter_slots(ggml_backend_buffer_t * dsts, int nslots, size_t dst_bytes,
        ggml_backend_buffer_t src, const size_t * src_offsets, const int * dst_rows,
        const int * slot_n, size_t row_bytes);
// Many small host ranges into one device buffer, one fence. False leaves the buffer unchanged.
GGML_BACKEND_API bool ggml_vk_write_ranges(ggml_backend_buffer_t dst, const void ** srcs,
        const size_t * dst_offsets, const size_t * sizes, int n);

// One graphics-queue copy into an exclusive device buffer, then one acquire on
// the compute queue. The caller runs the shader only after this returns.
GGML_BACKEND_API bool ggml_vk_upload_acquired(ggml_backend_buffer_t dst, size_t dst_offset, const void * src, size_t bytes);
// Submit the copy and return. Finish waits, acquires, and frees the ticket.
GGML_BACKEND_API void * ggml_vk_upload_begin(ggml_backend_buffer_t dst, size_t dst_offset, const void * src, size_t bytes);
GGML_BACKEND_API void ggml_vk_upload_finish(void * ticket);

struct ggml_vk_copy_region { size_t src_offset; size_t dst_offset; size_t bytes; };
// Same acquire, many regions, source already a Vulkan buffer. One fence.
GGML_BACKEND_API bool ggml_vk_copy_regions_acquired(ggml_backend_buffer_t dst, ggml_backend_buffer_t src,
        const struct ggml_vk_copy_region * regions, int n);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
