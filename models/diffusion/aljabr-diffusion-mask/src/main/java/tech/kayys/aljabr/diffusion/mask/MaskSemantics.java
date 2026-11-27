package tech.kayys.aljabr.diffusion.mask;

/** Describes whether mask values are discrete labels or continuous weights. */
public enum MaskSemantics {
    /** Each value is exactly zero or one. */
    BINARY,
    /** Each value is a continuous weight in the inclusive range [0, 1]. */
    SOFT
}
