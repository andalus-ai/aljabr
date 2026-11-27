package tech.kayys.aljabr.diffusion.mask;

/** Interpolation policy a mask transform must honor when resizing. */
public enum MaskResampling {
    /** Selects source labels without blending; required for discrete masks. */
    NEAREST_NEIGHBOR,
    /** Interpolates continuous mask weights; intended for soft masks. */
    BILINEAR
}
