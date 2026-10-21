/**
 * aljabr_metal_mps_cache.m — cached MPS descriptors and kernels.
 */

#import "aljabr_metal_mps_cache.h"
#import "aljabr_metal_support.h"

static BOOL g_disable_mps_cache = NO;

// MPSMatrixMultiplication and MPSMatrixDescriptor instances are immutable for a
// fixed shape/layout. Reusing them avoids rebuilding MPS state dozens of times
// per generated token while keeping Java-owned tensor memory zero-copy.
static NSMutableDictionary<NSString *, MPSMatrixMultiplication *>
    *g_mmul_cache = nil;
static NSMutableDictionary<NSString *, MPSMatrixDescriptor *>
    *g_matrix_desc_cache = nil;
static NSMutableDictionary<NSString *, MPSMatrixVectorMultiplication *>
    *g_mvec_cache = nil;
static NSMutableDictionary<NSString *, MPSVectorDescriptor *>
    *g_vector_desc_cache = nil;

void aljabr_metal_mps_cache_init(BOOL disabled) {
  g_disable_mps_cache = disabled;
  g_mmul_cache = [[NSMutableDictionary alloc] init];
  g_matrix_desc_cache = [[NSMutableDictionary alloc] init];
  g_mvec_cache = [[NSMutableDictionary alloc] init];
  g_vector_desc_cache = [[NSMutableDictionary alloc] init];
}

MPSMatrixDescriptor *aljabr_metal_cached_matrix_descriptor(
    int rows, int cols, NSUInteger row_bytes, MPSDataType data_type) {
  if (g_disable_mps_cache) {
    return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                 columns:cols
                                                rowBytes:row_bytes
                                                dataType:data_type];
  }
  NSString *key = [NSString stringWithFormat:@"%d:%d:%lu:%lu", rows, cols,
                                             (unsigned long)row_bytes,
                                             (unsigned long)data_type];
  @synchronized([MPSMatrixDescriptor class]) {
    if (g_matrix_desc_cache == nil) {
      g_matrix_desc_cache = [[NSMutableDictionary alloc] init];
    }
    MPSMatrixDescriptor *desc = [g_matrix_desc_cache objectForKey:key];
    if (desc == nil) {
      desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                   columns:cols
                                                  rowBytes:row_bytes
                                                  dataType:data_type];
      [g_matrix_desc_cache setObject:desc forKey:key];
    }
    return desc;
  }
}

MPSVectorDescriptor *
aljabr_metal_cached_vector_descriptor(int length, MPSDataType data_type) {
  if (g_disable_mps_cache) {
    return [MPSVectorDescriptor vectorDescriptorWithLength:length
                                                  dataType:data_type];
  }
  NSString *key =
      [NSString stringWithFormat:@"%d:%lu", length, (unsigned long)data_type];
  @synchronized([MPSVectorDescriptor class]) {
    if (g_vector_desc_cache == nil) {
      g_vector_desc_cache = [[NSMutableDictionary alloc] init];
    }
    MPSVectorDescriptor *desc = [g_vector_desc_cache objectForKey:key];
    if (desc == nil) {
      desc = [MPSVectorDescriptor vectorDescriptorWithLength:length
                                                    dataType:data_type];
      [g_vector_desc_cache setObject:desc forKey:key];
    }
    return desc;
  }
}

MPSMatrixMultiplication *aljabr_metal_cached_mmul(
    BOOL transpose_left, BOOL transpose_right, int result_rows, int result_cols,
    int interior_cols, MPSDataType left_type, MPSDataType right_type,
    MPSDataType result_type, float alpha, float beta) {
  if (g_disable_mps_cache) {
    return [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                             transposeLeft:transpose_left
                                            transposeRight:transpose_right
                                                resultRows:result_rows
                                             resultColumns:result_cols
                                           interiorColumns:interior_cols
                                                     alpha:alpha
                                                      beta:beta];
  }
  NSString *key = [NSString
      stringWithFormat:@"%d:%d:%d:%d:%d:%lu:%lu:%lu:%.8g:%.8g",
                       transpose_left ? 1 : 0, transpose_right ? 1 : 0,
                       result_rows, result_cols, interior_cols,
                       (unsigned long)left_type, (unsigned long)right_type,
                       (unsigned long)result_type, alpha, beta];
  @synchronized([MPSMatrixMultiplication class]) {
    if (g_mmul_cache == nil) {
      g_mmul_cache = [[NSMutableDictionary alloc] init];
    }
    MPSMatrixMultiplication *mmul = [g_mmul_cache objectForKey:key];
    if (mmul == nil) {
      mmul = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                               transposeLeft:transpose_left
                                              transposeRight:transpose_right
                                                  resultRows:result_rows
                                               resultColumns:result_cols
                                             interiorColumns:interior_cols
                                                       alpha:alpha
                                                        beta:beta];
      mmul.options = MPSKernelOptionsSkipAPIValidation;
      [g_mmul_cache setObject:mmul forKey:key];
    }
    return mmul;
  }
}

MPSMatrixVectorMultiplication *
aljabr_metal_cached_mvec(BOOL transpose, int rows, int columns,
                              double alpha, double beta) {
  if (g_disable_mps_cache) {
    return [[MPSMatrixVectorMultiplication alloc] initWithDevice:g_device
                                                       transpose:transpose
                                                            rows:rows
                                                         columns:columns
                                                           alpha:alpha
                                                            beta:beta];
  }
  NSString *key =
      [NSString stringWithFormat:@"%d:%d:%d:%.8g:%.8g", transpose ? 1 : 0, rows,
                                 columns, alpha, beta];
  @synchronized([MPSMatrixVectorMultiplication class]) {
    if (g_mvec_cache == nil) {
      g_mvec_cache = [[NSMutableDictionary alloc] init];
    }
    MPSMatrixVectorMultiplication *mvec = [g_mvec_cache objectForKey:key];
    if (mvec == nil) {
      mvec = [[MPSMatrixVectorMultiplication alloc] initWithDevice:g_device
                                                         transpose:transpose
                                                              rows:rows
                                                           columns:columns
                                                             alpha:alpha
                                                              beta:beta];
      [g_mvec_cache setObject:mvec forKey:key];
    }
    return mvec;
  }
}
