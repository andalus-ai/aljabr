/**
 * aljabr_metal_support.m — shared state and low-level helpers for the
 * Metal dylib.
 */

#import "aljabr_metal_support.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
BOOL g_initialized = NO;

BOOL aljabr_metal_env_truthy(const char *name) {
  const char *value = getenv(name);
  return value != NULL &&
         (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
          strcasecmp(value, "yes") == 0);
}

int aljabr_metal_env_int_or_default(const char *name, int default_value) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return default_value;
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value)
    return default_value;
  if (parsed < 0)
    return default_value;
  if (parsed > INT32_MAX)
    return INT32_MAX;
  return (int)parsed;
}

float aljabr_metal_env_float_or_default(const char *name,
                                             float default_value) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return default_value;
  char *end = NULL;
  float parsed = strtof(value, &end);
  if (end == value || !isfinite(parsed))
    return default_value;
  return parsed;
}

uint64_t aljabr_metal_monotonic_nanos(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static int g_metal_verbose = -1;

void aljabr_metal_set_verbose(int enable) {
  g_metal_verbose = enable ? 1 : 0;
}

BOOL aljabr_metal_is_verbose(void) {
  if (g_metal_verbose >= 0) {
    return g_metal_verbose == 1;
  }
  return aljabr_metal_env_truthy("ALJABR_METAL_VERBOSE") ||
         aljabr_metal_env_truthy("ALJABR_METAL_DEBUG") ||
         aljabr_metal_env_truthy("ALJABR_DEBUG") ||
         aljabr_metal_env_truthy("AQLI_DEBUG");
}
