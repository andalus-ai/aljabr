/**
 * aljabr_metal_mps_matvec_policy.m — MPS matvec validation/autotune
 * policy.
 */

#import "aljabr_metal_mps_matvec_policy.h"
#import "aljabr_metal_support.h"

static NSObject *g_policy_lock = nil;
static NSMutableSet<NSString *> *g_validated_shapes = nil;
static NSMutableSet<NSString *> *g_failed_shapes = nil;
static NSMutableSet<NSString *> *g_mps_preferred_shapes = nil;
static NSMutableSet<NSString *> *g_custom_preferred_shapes = nil;
static BOOL g_enable_mps_matvec = NO;
static BOOL g_enable_mps_matvec_autotune = NO;
static int g_max_inner_override = -1;
static int g_max_output_override = -1;
static int g_autotune_max_output_override = -1;
static BOOL g_disable_after_validation_failure = NO;

id aljabr_metal_mps_matvec_policy_lock(void) {
  if (g_policy_lock == nil) {
    g_policy_lock = [[NSObject alloc] init];
  }
  return g_policy_lock;
}

void aljabr_metal_mps_matvec_policy_init(void) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_validated_shapes = [[NSMutableSet alloc] init];
    g_failed_shapes = [[NSMutableSet alloc] init];
    g_mps_preferred_shapes = [[NSMutableSet alloc] init];
    g_custom_preferred_shapes = [[NSMutableSet alloc] init];
    g_disable_after_validation_failure = NO;
  }
}

int aljabr_metal_mps_matvec_set_enabled(int enabled) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_enable_mps_matvec = enabled != 0;
  }
  return 0;
}

int aljabr_metal_mps_matvec_set_autotune_enabled(int enabled) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_enable_mps_matvec_autotune = enabled != 0;
    [g_mps_preferred_shapes removeAllObjects];
    [g_custom_preferred_shapes removeAllObjects];
  }
  return 0;
}

int aljabr_metal_mps_matvec_set_max_inner(int max_inner) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_max_inner_override = max_inner;
  }
  return 0;
}

int aljabr_metal_mps_matvec_set_max_output(int max_output) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_max_output_override = max_output;
  }
  return 0;
}

int aljabr_metal_mps_matvec_set_autotune_max_output(int max_output) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_autotune_max_output_override = max_output;
  }
  return 0;
}

static BOOL base_should_try(int K, int N) {
  BOOL enabled = NO;
  BOOL disabledAfterFailure = NO;
  int maxInnerOverride = -1;
  int maxOutputOverride = -1;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    enabled = g_enable_mps_matvec;
    disabledAfterFailure = g_disable_after_validation_failure;
    maxInnerOverride = g_max_inner_override;
    maxOutputOverride = g_max_output_override;
  }

  int maxOutput = maxOutputOverride >= 0
                      ? maxOutputOverride
                      : aljabr_metal_env_int_or_default(
                            "ALJABR_METAL_MPS_MATVEC_MAX_OUTPUT", 640);
  int maxInner = maxInnerOverride >= 0
                     ? maxInnerOverride
                     : aljabr_metal_env_int_or_default(
                           "ALJABR_METAL_MPS_MATVEC_MAX_INNER", 2048);

  return !disabledAfterFailure &&
         (enabled || aljabr_metal_env_truthy(
                         "ALJABR_METAL_ENABLE_MPS_MATVEC")) &&
         (maxInner <= 0 || K <= maxInner) &&
         (maxOutput <= 0 || N <= maxOutput) &&
         !aljabr_metal_env_truthy("ALJABR_METAL_DISABLE_MPS_MATVEC");
}

BOOL aljabr_metal_mps_matvec_should_try(int K, int N) {
  return base_should_try(K, N);
}

BOOL aljabr_metal_mps_bf16_matvec_should_try(int K, int N) {
  return base_should_try(K, N)
         // Apple's MPSMatrixVector BF16 path can fail via fatal assertions
         // instead of catchable errors on current macOS SDKs. Keep it
         // unsafe lab opt-in; the custom BF16 Metal kernel is the safe default.
         && aljabr_metal_env_truthy(
                "ALJABR_METAL_UNSAFE_ENABLE_MPS_BF16_MATVEC") &&
         !aljabr_metal_env_truthy(
             "ALJABR_METAL_DISABLE_MPS_BF16_MATVEC");
}

BOOL aljabr_metal_mps_matvec_validate_every_call(void) {
  return aljabr_metal_env_truthy(
      "ALJABR_METAL_MPS_MATVEC_VALIDATE_EVERY_CALL");
}

BOOL aljabr_metal_mps_matvec_autotune_enabled_for_output(int N) {
  BOOL enabled = NO;
  int maxOutputOverride = -1;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    enabled = g_enable_mps_matvec_autotune;
    maxOutputOverride = g_autotune_max_output_override;
  }
  int maxOutput =
      maxOutputOverride >= 0
          ? maxOutputOverride
          : aljabr_metal_env_int_or_default(
                "ALJABR_METAL_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT", 1024);
  return (enabled || aljabr_metal_env_truthy(
                         "ALJABR_METAL_MPS_MATVEC_AUTOTUNE")) &&
         (maxOutput <= 0 || N <= maxOutput);
}

AljabrMetalMpsMatvecShapeState aljabr_metal_mps_matvec_shape_state(
    NSString *shapeKey, BOOL validateEveryCall, BOOL autotune) {
  AljabrMetalMpsMatvecShapeState state = {NO, NO, NO, NO};
  if (shapeKey == nil) {
    return state;
  }
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    state.failed =
        g_failed_shapes != nil && [g_failed_shapes containsObject:shapeKey];
    state.validated = !validateEveryCall && g_validated_shapes != nil &&
                      [g_validated_shapes containsObject:shapeKey];
    state.mps_preferred = autotune && g_mps_preferred_shapes != nil &&
                          [g_mps_preferred_shapes containsObject:shapeKey];
    state.custom_preferred =
        autotune && g_custom_preferred_shapes != nil &&
        [g_custom_preferred_shapes containsObject:shapeKey];
  }
  return state;
}

void aljabr_metal_mps_matvec_mark_validated(NSString *shapeKey) {
  if (shapeKey == nil)
    return;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    [g_validated_shapes addObject:shapeKey];
  }
}

void aljabr_metal_mps_matvec_mark_failed(NSString *shapeKey) {
  if (shapeKey == nil)
    return;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    [g_failed_shapes addObject:shapeKey];
  }
}

void aljabr_metal_mps_matvec_mark_mps_preferred(NSString *shapeKey) {
  if (shapeKey == nil)
    return;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    [g_mps_preferred_shapes addObject:shapeKey];
  }
}

void aljabr_metal_mps_matvec_record_autotune_preference(NSString *shapeKey,
                                                             BOOL preferMps) {
  if (shapeKey == nil)
    return;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    if (preferMps) {
      [g_mps_preferred_shapes addObject:shapeKey];
    } else {
      [g_custom_preferred_shapes addObject:shapeKey];
    }
  }
}

void aljabr_metal_mps_matvec_mark_disable_after_failure(void) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_disable_after_validation_failure = YES;
  }
}

AljabrMetalMpsMatvecOverrideSnapshot
aljabr_metal_mps_matvec_force_shape(int K, int N) {
  AljabrMetalMpsMatvecOverrideSnapshot snapshot;
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    snapshot.enabled = g_enable_mps_matvec;
    snapshot.max_inner_override = g_max_inner_override;
    snapshot.max_output_override = g_max_output_override;
    g_enable_mps_matvec = YES;
    g_max_inner_override = K;
    g_max_output_override = N;
  }
  return snapshot;
}

void aljabr_metal_mps_matvec_restore_overrides(
    AljabrMetalMpsMatvecOverrideSnapshot snapshot) {
  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    g_enable_mps_matvec = snapshot.enabled;
    g_max_inner_override = snapshot.max_inner_override;
    g_max_output_override = snapshot.max_output_override;
  }
}
