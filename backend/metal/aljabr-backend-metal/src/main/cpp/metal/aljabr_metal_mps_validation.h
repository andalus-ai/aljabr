/**
 * aljabr_metal_mps_validation.h — MPS matvec validation helpers.
 */

#ifndef ALJABR_METAL_MPS_VALIDATION_H
#define ALJABR_METAL_MPS_VALIDATION_H

#import <Foundation/Foundation.h>
#include <stdint.h>

NSString *aljabr_metal_mps_matvec_shape_key(int K, int N);
float aljabr_metal_bf16_to_f32(uint16_t bits);
uint16_t aljabr_metal_f32_to_bf16_bits(float value);

BOOL aljabr_metal_validate_mps_matvec_half_output(const float *C,
                                                       const float *A,
                                                       const uint16_t *B, int K,
                                                       int N);

BOOL aljabr_metal_validate_mps_matvec_bf16_output(const float *C,
                                                       const float *A,
                                                       const uint16_t *B, int K,
                                                       int N);

#define mps_matvec_shape_key aljabr_metal_mps_matvec_shape_key
#define bf16_to_f32_bridge aljabr_metal_bf16_to_f32
#define f32_to_bf16_bits_bridge aljabr_metal_f32_to_bf16_bits
#define validate_mps_matvec_half_output                                        \
  aljabr_metal_validate_mps_matvec_half_output
#define validate_mps_matvec_bf16_output                                        \
  aljabr_metal_validate_mps_matvec_bf16_output

#endif
