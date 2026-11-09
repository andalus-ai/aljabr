package tech.kayys.aljabr.backend.metal;

/** Policy for initializing the Metal runtime when native Metal is unavailable. */
public enum MetalRuntimePolicy {
    /** Preserve development/CI compatibility by allowing the Java CPU fallback. */
    ALLOW_CPU_FALLBACK,
    /** Fail construction unless the native Metal runtime initializes successfully. */
    REQUIRE_NATIVE
}
