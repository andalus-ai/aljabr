package tech.kayys.aljabr.models.flux;

/**
 * Enumeration of supported Black Forest Labs FLUX model variants.
 */
public enum FluxVariant {
    /**
     * FLUX.1 [schnell] - Fast 4-step distilled text-to-image model.
     */
    FLUX_1_SCHNELL("flux-1-schnell", 4, 1.0f, false),

    /**
     * FLUX.1 [dev] - High-quality guidance-distilled model (25-50 steps).
     */
    FLUX_1_DEV("flux-1-dev", 28, 3.5f, true),

    /**
     * FLUX.2-klein-9B - Compact high-performance 9B DiT variant by Black Forest Labs.
     */
    FLUX_2_KLEIN_9B("flux-2-klein-9b", 20, 3.5f, true);

    private final String modelCode;
    private final int defaultSteps;
    private final float defaultGuidanceScale;
    private final boolean requiresGuidanceEmbedding;

    FluxVariant(String modelCode, int defaultSteps, float defaultGuidanceScale, boolean requiresGuidanceEmbedding) {
        this.modelCode = modelCode;
        this.defaultSteps = defaultSteps;
        this.defaultGuidanceScale = defaultGuidanceScale;
        this.requiresGuidanceEmbedding = requiresGuidanceEmbedding;
    }

    public String modelCode() {
        return modelCode;
    }

    public int defaultSteps() {
        return defaultSteps;
    }

    public float defaultGuidanceScale() {
        return defaultGuidanceScale;
    }

    public boolean requiresGuidanceEmbedding() {
        return requiresGuidanceEmbedding;
    }
}
