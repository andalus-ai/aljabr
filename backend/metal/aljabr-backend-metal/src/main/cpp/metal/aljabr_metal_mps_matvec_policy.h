/**
 * aljabr_metal_mps_matvec_policy.h — MPS matvec validation/autotune
 * policy.
 */

#ifndef ALJABR_METAL_MPS_MATVEC_POLICY_H
#define ALJABR_METAL_MPS_MATVEC_POLICY_H

#import <Foundation/Foundation.h>

typedef struct {
  BOOL failed;
  BOOL validated;
  BOOL mps_preferred;
  BOOL custom_preferred;
} AljabrMetalMpsMatvecShapeState;

typedef struct {
  BOOL enabled;
  int max_inner_override;
  int max_output_override;
} AljabrMetalMpsMatvecOverrideSnapshot;

void aljabr_metal_mps_matvec_policy_init(void);
id aljabr_metal_mps_matvec_policy_lock(void);

int aljabr_metal_mps_matvec_set_enabled(int enabled);
int aljabr_metal_mps_matvec_set_autotune_enabled(int enabled);
int aljabr_metal_mps_matvec_set_max_inner(int max_inner);
int aljabr_metal_mps_matvec_set_max_output(int max_output);
int aljabr_metal_mps_matvec_set_autotune_max_output(int max_output);

BOOL aljabr_metal_mps_matvec_should_try(int K, int N);
BOOL aljabr_metal_mps_bf16_matvec_should_try(int K, int N);
BOOL aljabr_metal_mps_matvec_validate_every_call(void);
BOOL aljabr_metal_mps_matvec_autotune_enabled_for_output(int N);

AljabrMetalMpsMatvecShapeState
aljabr_metal_mps_matvec_shape_state(NSString *shapeKey,
                                         BOOL validateEveryCall, BOOL autotune);
void aljabr_metal_mps_matvec_mark_validated(NSString *shapeKey);
void aljabr_metal_mps_matvec_mark_failed(NSString *shapeKey);
void aljabr_metal_mps_matvec_mark_mps_preferred(NSString *shapeKey);
void aljabr_metal_mps_matvec_record_autotune_preference(NSString *shapeKey,
                                                             BOOL preferMps);
void aljabr_metal_mps_matvec_mark_disable_after_failure(void);

AljabrMetalMpsMatvecOverrideSnapshot
aljabr_metal_mps_matvec_force_shape(int K, int N);
void aljabr_metal_mps_matvec_restore_overrides(
    AljabrMetalMpsMatvecOverrideSnapshot snapshot);

#endif
