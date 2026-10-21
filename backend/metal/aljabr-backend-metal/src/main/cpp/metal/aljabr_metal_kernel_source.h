/**
 * aljabr_metal_kernel_source.h — runtime Metal shader source builders.
 */

#ifndef ALJABR_METAL_KERNEL_SOURCE_H
#define ALJABR_METAL_KERNEL_SOURCE_H

#import <Foundation/Foundation.h>

NSString *aljabr_metal_runtime_kernel_source(void);
NSString *aljabr_metal_matvec_kernel_source(NSUInteger threads);

#endif
