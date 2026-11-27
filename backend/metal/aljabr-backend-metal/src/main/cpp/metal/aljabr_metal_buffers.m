/**
 * aljabr_metal_buffers.m — shared Metal buffer helpers.
 *
 * Model-weight buffers use a CFMutableDictionaryRef cache. Ordinary tensor
 * pointer wrappers are scoped to individual synchronous dispatches so cached
 * Metal objects cannot outlive their Java FFM arena-backed storage.
 */

#import "aljabr_metal_buffers.h"
#import "aljabr_metal_support.h"
#import "aljabr_metal_runtime.h"
#import <CoreFoundation/CoreFoundation.h>
#import <os/lock.h>
#include <stdint.h>

static id<MTLBuffer> g_swiglu_gate_scratch = nil;
static id<MTLBuffer> g_swiglu_up_scratch = nil;
static id<MTLBuffer> g_swiglu_combined_scratch = nil;
static size_t g_swiglu_scratch_capacity = 0;
static size_t g_swiglu_combined_scratch_capacity = 0;

@interface AljabrCachedBuffer : NSObject
@property (nonatomic, strong) id<MTLBuffer> buffer;
@property (nonatomic, assign) size_t offset;
@end
@implementation AljabrCachedBuffer
@end

/* Ordinary tensor pointers are dispatch-scoped. Model weights use a separate
   cache because their backing storage has an explicit model lifetime. */
static CFMutableDictionaryRef g_weight_cache = NULL;
static dispatch_once_t g_cache_once;
static os_unfair_lock g_cache_lock = OS_UNFAIR_LOCK_INIT;

static void _ensure_cache(void) {
  dispatch_once(&g_cache_once, ^{
    g_weight_cache = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        NULL,
        &kCFTypeDictionaryValueCallBacks);
  });
}

static BOOL checked_aligned_region(uintptr_t address, size_t bytes,
                                    size_t page_size, uintptr_t *aligned_address,
                                    size_t *aligned_bytes, size_t *offset) {
  if (bytes == 0 || bytes > UINTPTR_MAX - address || page_size == 0 ||
      (page_size & (page_size - 1)) != 0)
    return NO;
  size_t page_offset = (size_t)(address & (uintptr_t)(page_size - 1));
  if (bytes > SIZE_MAX - page_offset)
    return NO;
  size_t span = bytes + page_offset;
  if (span > SIZE_MAX - (page_size - 1))
    return NO;
  size_t rounded = (span + page_size - 1) & ~(page_size - 1);
  if (rounded == 0 || address < page_offset)
    return NO;
  if (aligned_address)
    *aligned_address = address - page_offset;
  if (aligned_bytes)
    *aligned_bytes = rounded;
  if (offset)
    *offset = page_offset;
  return YES;
}

id<MTLBuffer> aljabr_metal_wrap_ptr(void *ptr, size_t bytes) {
  if (ptr == NULL || bytes == 0 || g_device == nil)
    return nil;

  /*
   * Tensor storage is owned by Java FFM arenas. A process-wide cache keyed by
   * raw pointers can outlive the arena and leave an MTLBuffer referencing
   * released memory (or a later allocation reusing the same address).
   *
   * Current Metal entry points submit and wait for their command buffers before
   * returning, so this per-dispatch wrapper remains valid for the complete GPU
   * use interval. Keep the model/weight cache separate: those buffers have an
   * explicit model lifetime and are not ordinary tensor storage.
   */
  id<MTLBuffer> wrapped = [g_device newBufferWithBytesNoCopy:ptr
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
  if (wrapped == nil) {
    /* No-copy Metal buffers require page-aligned pointer and length. */
    wrapped = [g_device newBufferWithBytes:ptr
                                    length:bytes
                                   options:MTLResourceStorageModeShared];
  }
  return wrapped;
}

#include <sys/mman.h>

static id<MTLBuffer> g_master_model_buffer = nil;
static uintptr_t g_master_base = 0;
static uintptr_t g_master_base_aligned = 0;
static size_t g_master_size = 0;

int aljabr_metal_register_model_buffer(const void *base_ptr, size_t total_bytes) {
  if (base_ptr == NULL || total_bytes == 0)
    return -1;
  if (!g_initialized)
    aljabr_metal_init();
  if (g_device == nil)
    return -1;

  uintptr_t addr = (uintptr_t)base_ptr;
  uintptr_t aligned_addr = 0;
  size_t aligned_bytes = 0;
  size_t offset = 0;
  if (!checked_aligned_region(addr, total_bytes, (size_t)getpagesize(),
                              &aligned_addr, &aligned_bytes, &offset))
    return -1;
  (void)offset;

  void *aligned_ptr = (void *)aligned_addr;
  id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:aligned_ptr
                                                  length:aligned_bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
  if (buf == nil)
    return -2;

  g_master_model_buffer = buf;
  g_master_base = addr;
  g_master_base_aligned = aligned_addr;
  g_master_size = total_bytes;
  madvise(aligned_ptr, aligned_bytes, MADV_WILLNEED);
  if (mlock(aligned_ptr, aligned_bytes) == 0) {
    ALJABR_METAL_LOG("[Metal Master Buffer] Locked %zu MB into physical RAM via mlock\n", aligned_bytes / (1024 * 1024));
  } else {
    ALJABR_METAL_LOG("[Metal Master Buffer] mlock advisory notice (%zu MB): %s\n", aligned_bytes / (1024 * 1024), strerror(errno));
  }

  // Pre-fault GPU page tables by committing a blit touching head and tail of the buffer.
  uint64_t t0 = aljabr_metal_monotonic_nanos();
  id<MTLBuffer> dummyDst = [g_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
  if (dummyDst != nil && g_queue != nil) {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    size_t touch_bytes = aligned_bytes < 64 ? aligned_bytes : 64;
    [blit copyFromBuffer:buf sourceOffset:0 toBuffer:dummyDst destinationOffset:0 size:touch_bytes];
    if (aligned_bytes > 128) {
      [blit copyFromBuffer:buf sourceOffset:aligned_bytes - touch_bytes toBuffer:dummyDst destinationOffset:0 size:touch_bytes];
    }
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
  uint64_t t1 = aljabr_metal_monotonic_nanos();

  ALJABR_METAL_LOG("[Metal Master Buffer] Registered and pre-faulted 0-copy MTLBuffer for model: %zu MB (aligned=%p, len=%zu MB) in %.2f ms\n",
                   total_bytes / (1024 * 1024), aligned_ptr, aligned_bytes / (1024 * 1024), (t1 - t0) * 1e-6);
  return 0;
}

id<MTLBuffer> aljabr_metal_wrap_weight_ptr(const void *ptr, size_t bytes) {
  size_t off = 0;
  return aljabr_metal_wrap_weight_ptr_offset(ptr, bytes, &off);
}

id<MTLBuffer> aljabr_metal_wrap_weight_ptr_offset(const void *ptr, size_t bytes, size_t *out_offset) {
  if (out_offset)
    *out_offset = 0;
  if (ptr == NULL || bytes == 0 || g_device == nil)
    return nil;

  uintptr_t addr = (uintptr_t)ptr;
  if (g_master_model_buffer != nil && addr >= g_master_base) {
    uintptr_t relative = addr - g_master_base;
    if (relative < g_master_size) {
      if (bytes > g_master_size - (size_t)relative)
        return nil;
      uintptr_t aligned_delta = addr - g_master_base_aligned;
      if (aligned_delta > (uintptr_t)SIZE_MAX)
        return nil;
      size_t master_offset = (size_t)aligned_delta;
      if (master_offset > g_master_model_buffer.length ||
          bytes > g_master_model_buffer.length - master_offset)
        return nil;
      if (out_offset)
        *out_offset = master_offset;
      return g_master_model_buffer;
    }
  }

  _ensure_cache();
  if (g_weight_cache == NULL)
    return nil;
  os_unfair_lock_lock(&g_cache_lock);
  AljabrCachedBuffer *cached =
      (__bridge AljabrCachedBuffer *)CFDictionaryGetValue(g_weight_cache, ptr);
  if (cached != nil) {
    size_t cached_offset = cached.offset;
    id<MTLBuffer> cached_buffer = cached.buffer;
    BOOL fits = cached_buffer != nil &&
                cached_offset <= cached_buffer.length &&
                bytes <= cached_buffer.length - cached_offset;
    if (fits) {
      if (out_offset)
        *out_offset = cached_offset;
      os_unfair_lock_unlock(&g_cache_lock);
      return cached_buffer;
    }
    CFDictionaryRemoveValue(g_weight_cache, ptr);
  }
  os_unfair_lock_unlock(&g_cache_lock);

  uintptr_t aligned_addr = 0;
  size_t aligned_bytes = 0;
  size_t offset = 0;
  if (!checked_aligned_region(addr, bytes, (size_t)getpagesize(),
                              &aligned_addr, &aligned_bytes, &offset))
    return nil;

  id<MTLBuffer> wrapped = [g_device newBufferWithBytesNoCopy:(void *)aligned_addr
                                                      length:aligned_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
  AljabrCachedBuffer *item = [AljabrCachedBuffer new];
  if (wrapped != nil) {
    item.buffer = wrapped;
    item.offset = offset;
  } else {
    wrapped = [g_device newBufferWithBytes:ptr
                                    length:bytes
                                   options:MTLResourceStorageModeShared];
    item.buffer = wrapped;
    item.offset = 0;
  }

  if (item.buffer == nil)
    return nil;
  os_unfair_lock_lock(&g_cache_lock);
  CFDictionarySetValue(g_weight_cache, ptr, (__bridge CFTypeRef)item);
  os_unfair_lock_unlock(&g_cache_lock);

  if (out_offset)
    *out_offset = item.offset;
  return item.buffer;
}

BOOL aljabr_metal_ensure_swiglu_scratch(size_t activation_bytes,
                                             id<MTLBuffer> *gate,
                                             id<MTLBuffer> *up,
                                             id<MTLBuffer> *combined) {
  if (activation_bytes == 0)
    return NO;
  if (g_swiglu_scratch_capacity < activation_bytes ||
      g_swiglu_gate_scratch == nil || g_swiglu_up_scratch == nil ||
      g_swiglu_combined_scratch == nil) {
    g_swiglu_gate_scratch =
        [g_device newBufferWithLength:activation_bytes
                              options:MTLResourceStorageModePrivate];
    g_swiglu_up_scratch =
        [g_device newBufferWithLength:activation_bytes
                              options:MTLResourceStorageModePrivate];
    g_swiglu_combined_scratch =
        [g_device newBufferWithLength:activation_bytes
                              options:MTLResourceStorageModePrivate];
    if (g_swiglu_gate_scratch == nil || g_swiglu_up_scratch == nil ||
        g_swiglu_combined_scratch == nil) {
      g_swiglu_gate_scratch = nil;
      g_swiglu_up_scratch = nil;
      g_swiglu_combined_scratch = nil;
      g_swiglu_scratch_capacity = 0;
      g_swiglu_combined_scratch_capacity = 0;
      return NO;
    }
    g_swiglu_scratch_capacity = activation_bytes;
    g_swiglu_combined_scratch_capacity = activation_bytes;
  }
  *gate = g_swiglu_gate_scratch;
  *up = g_swiglu_up_scratch;
  *combined = g_swiglu_combined_scratch;
  return YES;
}

BOOL aljabr_metal_ensure_swiglu_combined_scratch(size_t activation_bytes,
                                                      id<MTLBuffer> *combined) {
  if (activation_bytes == 0)
    return NO;
  if (g_swiglu_combined_scratch_capacity < activation_bytes ||
      g_swiglu_combined_scratch == nil) {
    g_swiglu_combined_scratch =
        [g_device newBufferWithLength:activation_bytes
                              options:MTLResourceStorageModePrivate];
    if (g_swiglu_combined_scratch == nil) {
      g_swiglu_combined_scratch_capacity = 0;
      return NO;
    }
    g_swiglu_combined_scratch_capacity = activation_bytes;
  }
  *combined = g_swiglu_combined_scratch;
  return YES;
}
