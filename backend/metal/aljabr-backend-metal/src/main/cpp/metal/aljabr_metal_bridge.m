/**
 * aljabr_metal_bridge.m — compatibility anchor for the Aljabr Metal dylib.
 *
 * The exported runtime, buffer, matvec, FFN, attention, and FA4 entry points
 * now live in focused modules next to this file. Keep this translation unit so
 * older build scripts that still mention aljabr_metal_bridge.m continue to
 * compile while the bridge stays intentionally small.
 */

#import "aljabr_metal_matvec_api.h"
