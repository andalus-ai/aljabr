/**
 * aljabr_metal_cpu_fallback.m — CPU fallback kernels for the Metal bridge.
 */

#import "aljabr_metal_cpu_fallback.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#ifndef ACCELERATE_LAPACK_ILP64
#define ACCELERATE_LAPACK_ILP64 1
#endif
#import <Accelerate/Accelerate.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static __thread float *tl_fallback_scratch0 = NULL;
static __thread float *tl_fallback_scratch1 = NULL;
static __thread size_t tl_fallback_scratch_capacity = 0;
static const int ALJABR_CPU_FFN_VECTOR_MIN_ELEMENTS = 2048;

static int ranges_overlap(const void *a, size_t a_bytes, const void *b,
                          size_t b_bytes) {
  uintptr_t a0 = (uintptr_t)a;
  uintptr_t b0 = (uintptr_t)b;
  uintptr_t a1 = a0 + a_bytes;
  uintptr_t b1 = b0 + b_bytes;
  return a0 < b1 && b0 < a1;
}

static int ensure_fallback_scratch(size_t elements, float **scratch0,
                                   float **scratch1) {
  if (scratch0 != NULL)
    *scratch0 = NULL;
  if (scratch1 != NULL)
    *scratch1 = NULL;
  if (elements == 0) {
    return 0;
  }
  if (tl_fallback_scratch_capacity < elements) {
    float *next0 = (float *)malloc(elements * sizeof(float));
    float *next1 = (float *)malloc(elements * sizeof(float));
    if (next0 == NULL || next1 == NULL) {
      free(next0);
      free(next1);
      return 0;
    }
    free(tl_fallback_scratch0);
    free(tl_fallback_scratch1);
    tl_fallback_scratch0 = next0;
    tl_fallback_scratch1 = next1;
    tl_fallback_scratch_capacity = elements;
  }
  if (scratch0 != NULL)
    *scratch0 = tl_fallback_scratch0;
  if (scratch1 != NULL)
    *scratch1 = tl_fallback_scratch1;
  return 1;
}

int aljabr_metal_cpu_add(void *C, const void *A, const void *B, int N) {
  const float *a = (const float *)A;
  const float *b = (const float *)B;
  float *c = (float *)C;
  size_t bytes = (size_t)N * sizeof(float);
  if (N >= 4096 && c == a && c == b) {
    float two = 2.0f;
    vDSP_vsmul(c, 1, &two, c, 1, (vDSP_Length)N);
    return 0;
  }
  if (N >= 4096 && c == a && !ranges_overlap(c, bytes, b, bytes)) {
    vDSP_vadd(c, 1, b, 1, c, 1, (vDSP_Length)N);
    return 0;
  }
  if (N >= 4096 && c == b && !ranges_overlap(c, bytes, a, bytes)) {
    vDSP_vadd(a, 1, c, 1, c, 1, (vDSP_Length)N);
    return 0;
  }
  if (N >= 4096 && !ranges_overlap(c, bytes, a, bytes) &&
      !ranges_overlap(c, bytes, b, bytes)) {
    vDSP_vadd(a, 1, b, 1, c, 1, (vDSP_Length)N);
    return 0;
  }
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
  return 0;
}

int aljabr_metal_cpu_matmul_nn(void *C, const void *A, const void *B,
                                    int M, int K, int N, float alpha,
                                    float beta) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              (const float *)A, K, (const float *)B, N, beta, (float *)C, N);
  return 0;
}

int aljabr_metal_cpu_matmul_tb(void *C, const void *A, const void *B,
                                    int M, int K, int N, float alpha,
                                    float beta) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
              (const float *)A, K, (const float *)B, K, beta, (float *)C, N);
  return 0;
}

int aljabr_metal_cpu_matvec_rows(void *y, const void *A, const void *x,
                                      int rows, int cols, int lda, float alpha,
                                      float beta) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, (const float *)A,
              lda, (const float *)x, 1, beta, (float *)y, 1);
  return 0;
}

int aljabr_metal_cpu_matvec_cols(void *y, const void *A, const void *x,
                                      int rows, int cols, int lda, float alpha,
                                      float beta) {
  cblas_sgemv(CblasRowMajor, CblasTrans, rows, cols, alpha, (const float *)A,
              lda, (const float *)x, 1, beta, (float *)y, 1);
  return 0;
}

int aljabr_metal_cpu_rmsnorm(void *out, const void *x, const void *weight,
                                  int N, float eps, int addOne) {
  const float *xi = (const float *)x;
  const float *wi = (const float *)weight;
  float *oi = (float *)out;
  float ss = 0;
  if (N >= 256) {
    vDSP_svesq(xi, 1, &ss, (vDSP_Length)N);
  } else {
    for (int i = 0; i < N; i++)
      ss += xi[i] * xi[i];
  }
  float inv = 1.0f / sqrtf(ss / N + eps);
  size_t bytes = (size_t)N * sizeof(float);
  if (!addOne && N >= 256 && !ranges_overlap(oi, bytes, xi, bytes) &&
      !ranges_overlap(oi, bytes, wi, bytes)) {
    vDSP_vmul(xi, 1, wi, 1, oi, 1, (vDSP_Length)N);
    vDSP_vsmul(oi, 1, &inv, oi, 1, (vDSP_Length)N);
    return 0;
  }
  for (int i = 0; i < N; i++) {
    float w = wi[i];
    if (addOne)
      w += 1.0f;
    oi[i] = xi[i] * inv * w;
  }
  return 0;
}

int aljabr_metal_cpu_silu_ffn(void *out, const void *gate, const void *up,
                                   int N) {
  const float *g = (const float *)gate;
  const float *u = (const float *)up;
  float *o = (float *)out;
  float *denom = NULL;
  float *numerator = NULL;
  if (N >= ALJABR_CPU_FFN_VECTOR_MIN_ELEMENTS &&
      ensure_fallback_scratch((size_t)N, &denom, &numerator)) {
    int count = N;
    float one = 1.0f;
    vDSP_vneg(g, 1, denom, 1, (vDSP_Length)N);
    vvexpf(denom, denom, &count);
    vDSP_vsadd(denom, 1, &one, denom, 1, (vDSP_Length)N);
    vDSP_vmul(g, 1, u, 1, numerator, 1, (vDSP_Length)N);
    // vDSP_vdiv intentionally takes denominator first, numerator second.
    vDSP_vdiv(denom, 1, numerator, 1, o, 1, (vDSP_Length)N);
    return 0;
  }
  for (int i = 0; i < N; i++) {
    float gi = g[i];
    o[i] = (gi / (1.0f + expf(-gi))) * u[i];
  }
  return 0;
}

int aljabr_metal_cpu_gelu_ffn(void *out, const void *gate, const void *up,
                                   int N) {
  const float *g = (const float *)gate;
  const float *u = (const float *)up;
  float *o = (float *)out;
  float *inner = NULL;
  float *scaled_gate_up = NULL;
  if (N >= ALJABR_CPU_FFN_VECTOR_MIN_ELEMENTS &&
      ensure_fallback_scratch((size_t)N, &inner, &scaled_gate_up)) {
    int count = N;
    float cubic_scale = 0.044715f;
    float gelu_scale = 0.79788456f;
    float half = 0.5f;
    float one = 1.0f;
    vDSP_vmul(g, 1, g, 1, inner, 1, (vDSP_Length)N);
    vDSP_vmul(inner, 1, g, 1, inner, 1, (vDSP_Length)N);
    vDSP_vsmsa(inner, 1, &cubic_scale, g, inner, 1, (vDSP_Length)N);
    vDSP_vsmul(inner, 1, &gelu_scale, inner, 1, (vDSP_Length)N);
    vvtanhf(inner, inner, &count);
    vDSP_vsadd(inner, 1, &one, inner, 1, (vDSP_Length)N);
    vDSP_vmul(g, 1, u, 1, scaled_gate_up, 1, (vDSP_Length)N);
    vDSP_vsmul(scaled_gate_up, 1, &half, scaled_gate_up, 1, (vDSP_Length)N);
    vDSP_vmul(scaled_gate_up, 1, inner, 1, o, 1, (vDSP_Length)N);
    return 0;
  }
  for (int i = 0; i < N; i++) {
    float gi = g[i];
    float inner = 0.79788456f * (gi + 0.044715f * gi * gi * gi);
    float gelu = 0.5f * gi * (1.0f + tanhf(inner));
    o[i] = gelu * u[i];
  }
  return 0;
}

int aljabr_metal_cpu_layernorm(void *out, const void *x,
                                    const void *weight, const void *bias, int N,
                                    float eps) {
  const float *xi = (const float *)x;
  const float *wi = (const float *)weight;
  const float *bi = (const float *)bias;
  float *oi = (float *)out;
  float s = 0;
  float ss = 0;
  for (int i = 0; i < N; i++) {
    float v = xi[i];
    s += v;
    ss += v * v;
  }
  float mean = s / N;
  float variance = (ss / N) - (mean * mean);
  float inv = 1.0f / sqrtf(variance + eps);
  for (int i = 0; i < N; i++) {
    float w = wi ? wi[i] : 1.0f;
    float b = bi ? bi[i] : 0.0f;
    oi[i] = (xi[i] - mean) * inv * w + b;
  }
  return 0;
}

int aljabr_metal_cpu_silu(void *out, const void *x, int N) {
  const float *xi = (const float *)x;
  float *oi = (float *)out;
  for (int i = 0; i < N; i++) {
    float g = xi[i];
    if (g <= -20.0f)
      oi[i] = 0.0f;
    else if (g >= 20.0f)
      oi[i] = g;
    else
      oi[i] = g / (1.0f + expf(-g));
  }
  return 0;
}

int aljabr_metal_cpu_gelu(void *out, const void *x, int N) {
  const float *xi = (const float *)x;
  float *oi = (float *)out;
  for (int i = 0; i < N; i++) {
    float g = xi[i];
    if (g <= -10.0f)
      oi[i] = 0.0f;
    else if (g >= 10.0f)
      oi[i] = g;
    else {
      float inner = 0.79788456f * (g + 0.044715f * g * g * g);
      oi[i] = 0.5f * g * (1.0f + tanhf(inner));
    }
  }
  return 0;
}
