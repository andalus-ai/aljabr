/**
 * aljabr_metal_runtime.h — exported runtime/device helpers for the Metal
 * dylib.
 */

#ifndef ALJABR_METAL_RUNTIME_H
#define ALJABR_METAL_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int aljabr_metal_init(void);
long aljabr_metal_available_memory(void);
int aljabr_metal_register_model_buffer(const void *base_ptr, size_t total_bytes);

int aljabr_metal_set_mps_matvec_enabled(int enabled);
int aljabr_metal_set_mps_matvec_autotune_enabled(int enabled);
int aljabr_metal_set_mps_matvec_max_inner(int max_inner);
int aljabr_metal_set_mps_matvec_max_output(int max_output);
int aljabr_metal_set_mps_matvec_autotune_max_output(int max_output);

void *aljabr_metal_alloc(size_t bytes, size_t align);

int aljabr_metal_argmax_f32(const void *logits, int n, int reject0,
                                 int reject1, int reject2, int reject3,
                                 int reject4, int reject5, int reject6,
                                 int reject7);

int aljabr_metal_device_name(char *buf, int bufSz);
int aljabr_metal_is_unified_memory(void);

#ifdef __cplusplus
}
#endif

#endif
