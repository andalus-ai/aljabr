/**
 * aljabr_metal_mps_validation.m — correctness checks for optional MPS
 * matvec paths.
 */

#import "aljabr_metal_mps_validation.h"
#import "aljabr_metal_scratch.h"
#import "aljabr_metal_support.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NSString *aljabr_metal_mps_matvec_shape_key(int K, int N) {
  return [NSString stringWithFormat:@"%d:%d", K, N];
}

static float half_matvec_reference_row(const float *A, const uint16_t *B, int K,
                                       int row) {
  const uint16_t *weight = B + ((size_t)row * (size_t)K);
  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += A[k] * aljabr_metal_f16_to_f32(weight[k]);
  }
  return sum;
}

float aljabr_metal_bf16_to_f32(uint16_t bits) {
  uint32_t raw = ((uint32_t)bits) << 16;
  float value;
  memcpy(&value, &raw, sizeof(value));
  return value;
}

uint16_t aljabr_metal_f32_to_bf16_bits(float value) {
  uint32_t raw;
  memcpy(&raw, &value, sizeof(raw));
  uint32_t lsb = (raw >> 16) & 1u;
  raw += 0x7FFFu + lsb;
  return (uint16_t)(raw >> 16);
}

static float bf16_matvec_reference_row(const float *A, const uint16_t *B, int K,
                                       int row) {
  const uint16_t *weight = B + ((size_t)row * (size_t)K);
  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += A[k] * aljabr_metal_bf16_to_f32(weight[k]);
  }
  return sum;
}

BOOL aljabr_metal_validate_mps_matvec_half_output(const float *C,
                                                       const float *A,
                                                       const uint16_t *B, int K,
                                                       int N) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
    return YES;
  }
  int samples = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_MPS_MATVEC_VALIDATE_SAMPLES", 8);
  if (samples <= 0) {
    samples = 1;
  }
  if (samples > N) {
    samples = N;
  }
  float abs_tol = 0.08f;
  float rel_tol = 0.03f;
  const char *abs_env = getenv("ALJABR_METAL_MPS_MATVEC_VALIDATE_ABS_TOL");
  const char *rel_env = getenv("ALJABR_METAL_MPS_MATVEC_VALIDATE_REL_TOL");
  if (abs_env != NULL && abs_env[0] != '\0')
    abs_tol = strtof(abs_env, NULL);
  if (rel_env != NULL && rel_env[0] != '\0')
    rel_tol = strtof(rel_env, NULL);

  // Deterministic spread over the vector. The custom Metal reduction kernel
  // remains the correctness fallback, so this check is allowed to be strict.
  uint32_t state = (uint32_t)(N * 2654435761u) ^ (uint32_t)(K * 2246822519u);
  for (int i = 0; i < samples; i++) {
    int row;
    if (i == 0) {
      row = 0;
    } else if (i == 1) {
      row = N - 1;
    } else if (i == 2) {
      row = N / 2;
    } else {
      state = state * 1664525u + 1013904223u;
      row = (int)(state % (uint32_t)N);
    }
    float actual = C[row];
    float expected = half_matvec_reference_row(A, B, K, row);
    if (!isfinite(actual) || !isfinite(expected)) {
      return NO;
    }
    float diff = fabsf(actual - expected);
    float limit = fmaxf(abs_tol, rel_tol * fmaxf(1.0f, fabsf(expected)));
    if (diff > limit) {
      if (aljabr_metal_env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
        fprintf(stderr,
                "[aljabr-metal] MPS matvec validation failed row=%d K=%d "
                "N=%d actual=%g expected=%g diff=%g limit=%g\n",
                row, K, N, actual, expected, diff, limit);
      }
      return NO;
    }
  }
  return YES;
}

BOOL aljabr_metal_validate_mps_matvec_bf16_output(const float *C,
                                                       const float *A,
                                                       const uint16_t *B, int K,
                                                       int N) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
    return YES;
  }
  int samples = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_MPS_MATVEC_VALIDATE_SAMPLES", 8);
  if (samples <= 0) {
    samples = 1;
  }
  if (samples > N) {
    samples = N;
  }
  float abs_tol = 0.12f;
  float rel_tol = 0.04f;
  const char *abs_env = getenv("ALJABR_METAL_MPS_MATVEC_VALIDATE_ABS_TOL");
  const char *rel_env = getenv("ALJABR_METAL_MPS_MATVEC_VALIDATE_REL_TOL");
  if (abs_env != NULL && abs_env[0] != '\0')
    abs_tol = strtof(abs_env, NULL);
  if (rel_env != NULL && rel_env[0] != '\0')
    rel_tol = strtof(rel_env, NULL);

  uint32_t state =
      (uint32_t)(N * 2654435761u) ^ (uint32_t)(K * 2246822519u) ^ 0x9E3779B9u;
  for (int i = 0; i < samples; i++) {
    int row;
    if (i == 0) {
      row = 0;
    } else if (i == 1) {
      row = N - 1;
    } else if (i == 2) {
      row = N / 2;
    } else {
      state = state * 1664525u + 1013904223u;
      row = (int)(state % (uint32_t)N);
    }
    float actual = C[row];
    float expected = bf16_matvec_reference_row(A, B, K, row);
    if (!isfinite(actual) || !isfinite(expected)) {
      return NO;
    }
    float diff = fabsf(actual - expected);
    float limit = fmaxf(abs_tol, rel_tol * fmaxf(1.0f, fabsf(expected)));
    if (diff > limit) {
      if (aljabr_metal_env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
        fprintf(stderr,
                "[aljabr-metal] MPS BF16 matvec validation mismatch "
                "row=%d actual=%f expected=%f diff=%f limit=%f K=%d N=%d\n",
                row, actual, expected, diff, limit, K, N);
      }
      return NO;
    }
  }
  return YES;
}
