#import "aljabr_metal_scratch.h"

#include <stdlib.h>
#include <string.h>

// Thread-local scratch buffers avoid malloc/free churn inside decode loops.
static __thread float *tl_k_scratch = NULL;
static __thread float *tl_v_scratch = NULL;
static __thread float *tl_score_scratch = NULL;
static __thread size_t tl_scratch_capacity = 0;
static __thread uint16_t *tl_half_input_scratch = NULL;
static __thread uint16_t *tl_half_output_scratch = NULL;
static __thread size_t tl_half_input_capacity = 0;
static __thread size_t tl_half_output_capacity = 0;

float aljabr_metal_f16_to_f32(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
  int exp = (int)((bits >> 10) & 0x1fu);
  uint32_t mant = bits & 0x03ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03ffu;
      uint32_t exp32 = (uint32_t)(exp + (127 - 15));
      out = sign | (exp32 << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    uint32_t exp32 = (uint32_t)(exp + (127 - 15));
    out = sign | (exp32 << 23) | (mant << 13);
  }
  float value;
  memcpy(&value, &out, sizeof(value));
  return value;
}

uint16_t aljabr_metal_f32_to_f16_bits(float value) {
  __fp16 half = (__fp16)value;
  uint16_t bits;
  memcpy(&bits, &half, sizeof(bits));
  return bits;
}

static float *aligned_float_alloc(size_t elements) {
  void *ptr = NULL;
  size_t bytes = elements * sizeof(float);
  if (posix_memalign(&ptr, 64, bytes) != 0) {
    return NULL;
  }
  return (float *)ptr;
}

int aljabr_metal_ensure_attention_scratch(size_t required_elements,
                                               float **k_scratch,
                                               float **v_scratch,
                                               float **score_scratch) {
  if (k_scratch != NULL)
    *k_scratch = NULL;
  if (v_scratch != NULL)
    *v_scratch = NULL;
  if (score_scratch != NULL)
    *score_scratch = NULL;
  if (required_elements == 0) {
    return 1;
  }
  if (tl_scratch_capacity < required_elements) {
    free(tl_k_scratch);
    free(tl_v_scratch);
    free(tl_score_scratch);

    size_t new_cap = required_elements + (required_elements / 4);
    tl_k_scratch = aligned_float_alloc(new_cap);
    tl_v_scratch = aligned_float_alloc(new_cap);
    tl_score_scratch = aligned_float_alloc(new_cap);
    if (!tl_k_scratch || !tl_v_scratch || !tl_score_scratch) {
      free(tl_k_scratch);
      free(tl_v_scratch);
      free(tl_score_scratch);
      tl_k_scratch = NULL;
      tl_v_scratch = NULL;
      tl_score_scratch = NULL;
      tl_scratch_capacity = 0;
      return 0;
    }
    tl_scratch_capacity = new_cap;
  }
  if (k_scratch != NULL)
    *k_scratch = tl_k_scratch;
  if (v_scratch != NULL)
    *v_scratch = tl_v_scratch;
  if (score_scratch != NULL)
    *score_scratch = tl_score_scratch;
  return 1;
}

static uint16_t *ensure_half_scratch(uint16_t **scratch, size_t *capacity,
                                     size_t required_elements) {
  if (required_elements == 0)
    return NULL;
  if (*capacity < required_elements || *scratch == NULL) {
    size_t new_cap = required_elements * 2;
    uint16_t *next = (uint16_t *)realloc(*scratch, new_cap * sizeof(uint16_t));
    if (next == NULL) {
      free(*scratch);
      *scratch = NULL;
      *capacity = 0;
      return NULL;
    }
    *scratch = next;
    *capacity = new_cap;
  }
  return *scratch;
}

uint16_t *
aljabr_metal_ensure_half_input_scratch(size_t required_elements) {
  return ensure_half_scratch(&tl_half_input_scratch, &tl_half_input_capacity,
                             required_elements);
}

uint16_t *
aljabr_metal_ensure_half_output_scratch(size_t required_elements) {
  return ensure_half_scratch(&tl_half_output_scratch, &tl_half_output_capacity,
                             required_elements);
}
