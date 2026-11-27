package tech.kayys.aljabr.diffusion.mask;

/** Immutable semantic and resampling policy for one mask adapter. */
public record MaskConditioningOptions(
        MaskSemantics semantics,
        MaskResampling resampling,
        float threshold) {
    public MaskConditioningOptions {
        if (semantics == null) throw new NullPointerException("semantics");
        if (resampling == null) throw new NullPointerException("resampling");
        if (!Float.isFinite(threshold) || threshold < 0f || threshold > 1f) {
            throw new IllegalArgumentException("threshold must be finite and in [0, 1]");
        }
        if (semantics == MaskSemantics.BINARY && resampling != MaskResampling.NEAREST_NEIGHBOR) {
            throw new IllegalArgumentException("binary masks require nearest-neighbor resampling");
        }
    }
    public static MaskConditioningOptions binary(float threshold) {
        return new MaskConditioningOptions(MaskSemantics.BINARY, MaskResampling.NEAREST_NEIGHBOR, threshold);
    }
    public static MaskConditioningOptions soft() {
        return new MaskConditioningOptions(MaskSemantics.SOFT, MaskResampling.BILINEAR, 0.5f);
    }
}
