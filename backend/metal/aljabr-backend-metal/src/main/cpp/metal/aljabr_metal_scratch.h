#ifndef ALJABR_METAL_SCRATCH_H
#define ALJABR_METAL_SCRATCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float aljabr_metal_f16_to_f32(uint16_t bits);
uint16_t aljabr_metal_f32_to_f16_bits(float value);

int aljabr_metal_ensure_attention_scratch(size_t required_elements,
                                               float **k_scratch,
                                               float **v_scratch,
                                               float **score_scratch);

uint16_t *aljabr_metal_ensure_half_input_scratch(size_t required_elements);
uint16_t *
aljabr_metal_ensure_half_output_scratch(size_t required_elements);

#ifdef __cplusplus
}
#endif

#define f16_to_f32 aljabr_metal_f16_to_f32
#define f32_to_f16_bits aljabr_metal_f32_to_f16_bits

#endif
