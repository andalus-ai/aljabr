/**
 * aljabr_metal_runtime.m — exported runtime/device helpers for the Metal
 * dylib.
 */

#import "aljabr_metal_elementwise.h"
#import "aljabr_metal_mps_cache.h"
#import "aljabr_metal_mps_matmul.h"
#import "aljabr_metal_mps_matvec_policy.h"
#import "aljabr_metal_pipelines.h"
#import "aljabr_metal_runtime.h"
#import "aljabr_metal_support.h"

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static NSMutableArray<id<MTLBuffer>> *g_external_allocations = nil;

static NSMutableArray<id<MTLBuffer>> *retained_external_allocations(void) {
  if (g_external_allocations == nil) {
    g_external_allocations = [[NSMutableArray alloc] init];
  }
  return g_external_allocations;
}

int aljabr_metal_init(void) {
  if (g_initialized)
    return 0;
  g_device = MTLCreateSystemDefaultDevice();
  if (!g_device)
    return -1;
  g_queue = [g_device newCommandQueue];
  aljabr_metal_mps_matmul_configure_from_env();
  aljabr_metal_mps_cache_init(
      aljabr_metal_env_truthy("ALJABR_METAL_DISABLE_MPS_CACHE"));
  aljabr_metal_mps_matvec_policy_init();
  BOOL enable_elementwise_kernels = aljabr_metal_env_truthy(
      "ALJABR_METAL_ENABLE_ELEMENTWISE_KERNELS");
  BOOL explicit_disable_elementwise = aljabr_metal_env_truthy(
      "ALJABR_METAL_DISABLE_ELEMENTWISE_KERNELS");
  aljabr_metal_elementwise_set_enabled(enable_elementwise_kernels &&
                                            !explicit_disable_elementwise);
  aljabr_metal_compile_runtime_pipelines(aljabr_metal_pipelines());
  g_initialized = YES;
  return 0;
}

long aljabr_metal_available_memory(void) {
  if (!g_device)
    return 0;
  return (long)[g_device recommendedMaxWorkingSetSize];
}

int aljabr_metal_set_mps_matvec_enabled(int enabled) {
  return aljabr_metal_mps_matvec_set_enabled(enabled);
}

int aljabr_metal_set_mps_matvec_autotune_enabled(int enabled) {
  return aljabr_metal_mps_matvec_set_autotune_enabled(enabled);
}

int aljabr_metal_set_mps_matvec_max_inner(int max_inner) {
  return aljabr_metal_mps_matvec_set_max_inner(max_inner);
}

int aljabr_metal_set_mps_matvec_max_output(int max_output) {
  return aljabr_metal_mps_matvec_set_max_output(max_output);
}

int aljabr_metal_set_mps_matvec_autotune_max_output(int max_output) {
  return aljabr_metal_mps_matvec_set_autotune_max_output(max_output);
}

void *aljabr_metal_alloc(size_t bytes, size_t align) {
  (void)align;
  if (!g_device || bytes == 0)
    return NULL;
  id<MTLBuffer> buf =
      [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (buf == nil)
    return NULL;
  NSMutableArray<id<MTLBuffer>> *allocations = retained_external_allocations();
  @synchronized(allocations) {
    [allocations addObject:buf];
  }
  return [buf contents];
}

static inline BOOL aljabr_metal_argmax_rejected(int id, int reject0,
                                                     int reject1, int reject2,
                                                     int reject3, int reject4,
                                                     int reject5, int reject6,
                                                     int reject7) {
  return id == reject0 || id == reject1 || id == reject2 || id == reject3 ||
         id == reject4 || id == reject5 || id == reject6 || id == reject7;
}

static int aljabr_metal_argmax_scalar_f32(const float *values, int n,
                                               int reject0, int reject1,
                                               int reject2, int reject3,
                                               int reject4, int reject5,
                                               int reject6, int reject7) {
  int best = -1;
  float best_val = -INFINITY;
  for (int i = 0; i < n; i++) {
    if (aljabr_metal_argmax_rejected(i, reject0, reject1, reject2, reject3,
                                          reject4, reject5, reject6, reject7)) {
      continue;
    }
    float value = values[i];
    if (isnan(value)) {
      continue;
    }
    if (value > best_val) {
      best_val = value;
      best = i;
    }
  }
  return best;
}

static int aljabr_metal_argmax_collect_rejections(
    int n, int out[8], int reject0, int reject1, int reject2, int reject3,
    int reject4, int reject5, int reject6, int reject7) {
  int raw[8] = {reject0, reject1, reject2, reject3,
                reject4, reject5, reject6, reject7};
  int count = 0;
  for (int i = 0; i < 8; i++) {
    int id = raw[i];
    if (id < 0 || id >= n) {
      continue;
    }
    BOOL duplicate = NO;
    for (int j = 0; j < count; j++) {
      if (out[j] == id) {
        duplicate = YES;
        break;
      }
    }
    if (!duplicate) {
      out[count++] = id;
    }
  }
  for (int i = 1; i < count; i++) {
    int value = out[i];
    int j = i - 1;
    while (j >= 0 && out[j] > value) {
      out[j + 1] = out[j];
      j--;
    }
    out[j + 1] = value;
  }
  return count;
}

static int aljabr_metal_argmax_vdsp_range_f32(const float *values,
                                                   int start, int end,
                                                   float *best_value,
                                                   int *best_index) {
  int length = end - start;
  if (length <= 0) {
    return 0;
  }
  float max_value = -INFINITY;
  vDSP_Length max_index = 0;
  vDSP_maxvi(values + start, 1, &max_value, &max_index, (vDSP_Length)length);
  if (isnan(max_value)) {
    return -1;
  }
  if (*best_index < 0 || max_value > *best_value) {
    *best_value = max_value;
    *best_index = start + (int)max_index;
  }
  return 0;
}

static int aljabr_metal_argmax_vdsp_f32(const float *values, int n,
                                             int reject0, int reject1,
                                             int reject2, int reject3,
                                             int reject4, int reject5,
                                             int reject6, int reject7) {
  int rejections[8] = {0};
  int rejection_count = aljabr_metal_argmax_collect_rejections(
      n, rejections, reject0, reject1, reject2, reject3, reject4, reject5,
      reject6, reject7);
  float best_value = -INFINITY;
  int best_index = -1;
  int start = 0;
  for (int i = 0; i < rejection_count; i++) {
    if (aljabr_metal_argmax_vdsp_range_f32(values, start, rejections[i],
                                                &best_value, &best_index) < 0) {
      return INT_MIN;
    }
    start = rejections[i] + 1;
  }
  if (aljabr_metal_argmax_vdsp_range_f32(values, start, n, &best_value,
                                              &best_index) < 0) {
    return INT_MIN;
  }
  return best_index;
}

int aljabr_metal_argmax_f32(const void *logits, int n, int reject0,
                                 int reject1, int reject2, int reject3,
                                 int reject4, int reject5, int reject6,
                                 int reject7) {
  if (logits == NULL || n <= 0)
    return -1;
  const float *values = (const float *)logits;
  int vdsp_best = aljabr_metal_argmax_vdsp_f32(values, n, reject0, reject1,
                                                    reject2, reject3, reject4,
                                                    reject5, reject6, reject7);
  if (vdsp_best != INT_MIN) {
    return vdsp_best;
  }
  return aljabr_metal_argmax_scalar_f32(values, n, reject0, reject1,
                                             reject2, reject3, reject4, reject5,
                                             reject6, reject7);
}

int aljabr_metal_device_name(char *buf, int bufSz) {
  if (!g_device || buf == NULL || bufSz <= 0)
    return -1;
  snprintf(buf, (size_t)bufSz, "%s", [[g_device name] UTF8String]);
  return 0;
}

int aljabr_metal_is_unified_memory(void) {
  return (g_device && [g_device hasUnifiedMemory]) ? 1 : 0;
}
