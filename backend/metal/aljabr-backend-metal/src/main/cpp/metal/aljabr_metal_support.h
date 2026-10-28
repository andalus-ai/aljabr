/**
 * aljabr_metal_support.h — shared state and low-level helpers for the
 * Metal dylib.
 */

#ifndef ALJABR_METAL_SUPPORT_H
#define ALJABR_METAL_SUPPORT_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;
extern BOOL g_initialized;

BOOL aljabr_metal_env_truthy(const char *name);
int aljabr_metal_env_int_or_default(const char *name, int default_value);
float aljabr_metal_env_float_or_default(const char *name,
                                             float default_value);
uint64_t aljabr_metal_monotonic_nanos(void);

void aljabr_metal_set_verbose(int enable);
BOOL aljabr_metal_is_verbose(void);

#define ALJABR_METAL_LOG(...) do { \
  if (aljabr_metal_is_verbose()) { \
    fprintf(stderr, __VA_ARGS__); \
  } \
} while(0)

#endif
