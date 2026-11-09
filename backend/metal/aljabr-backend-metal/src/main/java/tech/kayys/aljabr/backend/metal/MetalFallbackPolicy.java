package tech.kayys.aljabr.backend.metal;

/**
 * Controls whether Metal operations may transparently execute on the CPU.
 */
public enum MetalFallbackPolicy {
    /** Preserve compatibility by executing unsupported operations on the CPU backend. */
    ALLOW_CPU_FALLBACK,
    /** Reject operations that do not have a usable native Metal implementation. */
    FAIL_IF_UNSUPPORTED
}
