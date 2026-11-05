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
// Compute the graph, and record these device→host-visible copies at the end of
// its command buffer, after the shaders. After ggml_backend_synchronize,
// ggml_vk_take_suffix_reads memcpys them out. Not a host-mapped device buffer.
// n == 0 computes as usual.
GGML_BACKEND_API enum ggml_status ggml_vk_graph_compute_suffix(ggml_backend_t backend, struct ggml_cgraph * graph,
        const struct ggml_tensor ** tensors, const size_t * sizes, int n);
GGML_BACKEND_API bool ggml_vk_take_suffix_reads(ggml_backend_t backend, void ** dsts, const size_t * sizes, int n);
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
// Zero [offset, offset+bytes) then, if n > 0, copy ranges from src. Compute queue, one fence.
GGML_BACKEND_API bool ggml_vk_fill_and_copy(ggml_backend_buffer_t dst, size_t zero_off, size_t zero_bytes,
        ggml_backend_buffer_t src, const size_t * src_offsets, const size_t * dst_offsets, const size_t * sizes, int n);

// One graphics-queue copy into an exclusive device buffer, then one acquire on
// the compute queue. The caller runs the shader only after this returns.
GGML_BACKEND_API bool ggml_vk_upload_acquired(ggml_backend_buffer_t dst, size_t dst_offset, const void * src, size_t bytes);
// Submit the copy and return. Finish waits for the transfer and frees the ticket.
// Returns 1 if the compute queue can read the bytes now. Returns 0 if the
// acquire is deferred: *ready_flag stays 0 until ggml_vk_flush_deferred_acquires,
// which sets it to 1 on success or -1 on failure. A deferred acquire is not
// submitted on the compute queue, so it cannot sit in front of the next GPU sync.
GGML_BACKEND_API void * ggml_vk_upload_begin(ggml_backend_buffer_t dst, size_t dst_offset, const void * src, size_t bytes);
GGML_BACKEND_API int ggml_vk_upload_finish(void * ticket, int * ready_flag);
// Many host ranges, one transfer, one compute-queue acquire. Sets each ready
// flag to 1 on success or -1 on failure. Returns 1 if the compute queue can
// read the bytes. The caller must not reuse src pointers until this returns.
GGML_BACKEND_API int ggml_vk_upload_host_ranges(ggml_backend_buffer_t dst,
        const size_t * dst_offsets, const void * const * srcs, const size_t * sizes, int n,
        int ** ready_flags, int nflags);
GGML_BACKEND_API void ggml_vk_flush_deferred_acquires(void);
// One expert into a reused staging buffer. No compute-queue acquire, so this
// does not sit in front of the next dense sync. Returns 1 if the compute queue
// can read the bytes now, 2 if ggml_vk_decode_upload_acquire still owes a
// barrier, 0 on failure, -1 if the upload window closed before submit.
GGML_BACKEND_API int ggml_vk_upload_expert_fast(ggml_backend_buffer_t dst,
        size_t off0, const void * src0, size_t n0,
        size_t off1, const void * src1, size_t n1,
        size_t off2, const void * src2, size_t n2);
// One acquire for every fast upload that returned 2. Compute queue must be idle.
GGML_BACKEND_API int ggml_vk_decode_upload_acquire(void);
// Open only while the CPU is waiting on a GPU fence, before the miss walk.
GGML_BACKEND_API void ggml_vk_decode_upload_window(int open);
GGML_BACKEND_API int ggml_vk_decode_upload_window_open(void);

// Device buffer the CPU can read after the compute fence. Null if the device
// has no host-visible memory. ptr is the mapping of offset 0.
// Record these device copies at the start of the next compute command buffer
// on this backend, before its shaders. Same queue, so no extra fence.
GGML_BACKEND_API void ggml_vk_queue_copies(ggml_backend_t backend, ggml_backend_buffer_t dst,
        ggml_backend_buffer_t src, const struct ggml_vk_copy_region * regions, int n);

GGML_BACKEND_API ggml_backend_buffer_t ggml_vk_alloc_mapped(ggml_backend_t backend, size_t size);
GGML_BACKEND_API void * ggml_vk_mapped_ptr(ggml_backend_buffer_t buffer);

struct ggml_vk_copy_region { size_t src_offset; size_t dst_offset; size_t bytes; };
// Same acquire, many regions, source already a Vulkan buffer. One fence.
GGML_BACKEND_API bool ggml_vk_copy_regions_acquired(ggml_backend_buffer_t dst, ggml_backend_buffer_t src,
        const struct ggml_vk_copy_region * regions, int n);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// Record one decode graph into a side context. Does not submit.
GGML_BACKEND_API bool ggml_vk_park_record(ggml_backend_t backend, struct ggml_cgraph * graph);
GGML_BACKEND_API bool ggml_vk_park_submit(ggml_backend_t backend);
GGML_BACKEND_API void ggml_vk_park_drop(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
