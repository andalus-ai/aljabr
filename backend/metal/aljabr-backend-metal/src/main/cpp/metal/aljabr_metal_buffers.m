/**
 * aljabr_metal_buffers.m — shared Metal buffer helpers.
 *
 * Buffer cache uses CFMutableDictionaryRef with NULL key callbacks so raw
 * void* pointers are stored as opaque identity keys — no ObjC messaging on
 * the key, no ARC bridge-cast UB.
 */

#import "aljabr_metal_buffers.h"
#import "aljabr_metal_support.h"
#import "aljabr_metal_runtime.h"
#import <CoreFoundation/CoreFoundation.h>
#import <os/lock.h>

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

/* CFDictionary with NULL key callbacks: keys compared by pointer identity,
   no retain/release on keys.  Values use kCFTypeDictionaryValueCallBacks so
   the MTLBuffer objects are retained properly. */
static CFMutableDictionaryRef g_ptr_buffer_cache = NULL;
static CFMutableDictionaryRef g_weight_cache = NULL;
static dispatch_once_t g_cache_once;
static os_unfair_lock g_cache_lock = OS_UNFAIR_LOCK_INIT;

static void _ensure_cache(void) {
  dispatch_once(&g_cache_once, ^{
    /* NULL key callbacks = pointer-identity keys, no memory management */
    g_ptr_buffer_cache = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        NULL,                          /* key callbacks: NULL = identity */
        &kCFTypeDictionaryValueCallBacks); /* value callbacks: retain/release */
    g_weight_cache = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        NULL,
        &kCFTypeDictionaryValueCallBacks);
  });
}

id<MTLBuffer> aljabr_metal_wrap_ptr(void *ptr, size_t bytes) {
  if (ptr == NULL || bytes == 0)
    return nil;
  if (aljabr_metal_env_truthy("ALJABR_METAL_DISABLE_BUFFER_CACHE")) {
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
  }
  _ensure_cache();
  os_unfair_lock_lock(&g_cache_lock);
  id<MTLBuffer> cached = (__bridge id<MTLBuffer>)CFDictionaryGetValue(g_ptr_buffer_cache, ptr);
  if (cached != nil && cached.length >= bytes) {
    os_unfair_lock_unlock(&g_cache_lock);
    return cached;
  }
  os_unfair_lock_unlock(&g_cache_lock);

  /* Allocate outside the lock — newBufferWithBytesNoCopy is thread-safe */
  id<MTLBuffer> wrapped = [g_device newBufferWithBytesNoCopy:ptr
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
  if (wrapped == nil) {
    /* If pointer or length is not page-aligned, fall back to copy buffer */
    wrapped = [g_device newBufferWithBytes:ptr
                                    length:bytes
                                   options:MTLResourceStorageModeShared];
  }
  if (wrapped != nil) {
    os_unfair_lock_lock(&g_cache_lock);
    /* Check again in case another thread raced us */
    id<MTLBuffer> existing = (__bridge id<MTLBuffer>)CFDictionaryGetValue(g_ptr_buffer_cache, ptr);
    if (existing == nil || existing.length < bytes) {
      CFDictionarySetValue(g_ptr_buffer_cache, ptr, (__bridge CFTypeRef)wrapped);
    }
    os_unfair_lock_unlock(&g_cache_lock);
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
  if (!g_initialized) {
    aljabr_metal_init();
  }
  if (g_device == nil)
    return -1;

  size_t page_size = (size_t)getpagesize();
  uintptr_t addr = (uintptr_t)base_ptr;
  uintptr_t offset = addr & (page_size - 1);
  void *aligned_ptr = (void *)(addr - offset);
  size_t aligned_bytes = (total_bytes + offset + page_size - 1) & ~(page_size - 1);

  id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:aligned_ptr
                                                  length:aligned_bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
  if (buf == nil) {
    fprintf(stderr, "[Metal Master Buffer] Failed to allocate master buffer (%zu MB)\n", aligned_bytes / (1024 * 1024));
    return -2;
  }
  g_master_model_buffer = buf;
  g_master_base = addr;
  g_master_base_aligned = (uintptr_t)aligned_ptr;
  g_master_size = total_bytes;
  madvise(aligned_ptr, aligned_bytes, MADV_WILLNEED);

  // Pre-fault GPU page tables by committing a blit touching head and tail of the buffer.
  // This moves the one-time IOGPU driver page-table mapping (5+ seconds for 6.8 GB)
  // from Layer 0 prefill into model load time.
  uint64_t t0 = aljabr_metal_monotonic_nanos();
  id<MTLBuffer> dummyDst = [g_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
  if (dummyDst != nil && g_queue != nil) {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:buf sourceOffset:0 toBuffer:dummyDst destinationOffset:0 size:64];
    if (aligned_bytes > 128) {
      [blit copyFromBuffer:buf sourceOffset:aligned_bytes - 64 toBuffer:dummyDst destinationOffset:0 size:64];
    }
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
  uint64_t t1 = aljabr_metal_monotonic_nanos();

  printf("[Metal Master Buffer] Registered and pre-faulted 0-copy MTLBuffer for model: %zu MB (aligned=%p, len=%zu MB) in %.2f ms\n",
         total_bytes / (1024 * 1024), aligned_ptr, aligned_bytes / (1024 * 1024), (t1 - t0) * 1e-6);
  return 0;
}

id<MTLBuffer> aljabr_metal_wrap_weight_ptr(const void *ptr, size_t bytes) {
  size_t off = 0;
  return aljabr_metal_wrap_weight_ptr_offset(ptr, bytes, &off);
}

id<MTLBuffer> aljabr_metal_wrap_weight_ptr_offset(const void *ptr, size_t bytes, size_t *out_offset) {
  if (ptr == NULL || bytes == 0) {
    if (out_offset) *out_offset = 0;
    return nil;
  }

  uintptr_t addr = (uintptr_t)ptr;
  if (g_master_model_buffer != nil &&
      addr >= g_master_base &&
      (addr + bytes) <= (g_master_base + g_master_size)) {
    if (out_offset) {
      *out_offset = (size_t)(addr - g_master_base_aligned);
    }
    return g_master_model_buffer;
  }

  _ensure_cache();
  os_unfair_lock_lock(&g_cache_lock);
  AljabrCachedBuffer *cached = (__bridge AljabrCachedBuffer *)CFDictionaryGetValue(g_weight_cache, ptr);
  if (cached != nil) {
    if (out_offset) *out_offset = cached.offset;
    id<MTLBuffer> b = cached.buffer;
    os_unfair_lock_unlock(&g_cache_lock);
    return b;
  }
  os_unfair_lock_unlock(&g_cache_lock);

  size_t page_size = (size_t)getpagesize();
  uintptr_t fallback_addr = (uintptr_t)ptr;
  uintptr_t offset = fallback_addr & (page_size - 1);
  void *aligned_ptr = (void *)(fallback_addr - offset);
  size_t aligned_bytes = (bytes + offset + page_size - 1) & ~(page_size - 1);

  id<MTLBuffer> wrapped = [g_device newBufferWithBytesNoCopy:aligned_ptr
                                                      length:aligned_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
  AljabrCachedBuffer *item = [AljabrCachedBuffer new];
  if (wrapped != nil) {
    item.buffer = wrapped;
    item.offset = offset;
  } else {
    wrapped = [g_device newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
    item.buffer = wrapped;
    item.offset = 0;
  }

  if (item.buffer != nil) {
    os_unfair_lock_lock(&g_cache_lock);
    CFDictionarySetValue(g_weight_cache, ptr, (__bridge CFTypeRef)item);
    os_unfair_lock_unlock(&g_cache_lock);
  }

  if (out_offset) *out_offset = item.offset;
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
