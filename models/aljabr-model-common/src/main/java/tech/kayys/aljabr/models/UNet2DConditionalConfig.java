package tech.kayys.aljabr.models;

/**
 * Immutable architecture dimensions for the current two-resolution conditional UNet.
 *
 * <p>The configuration controls channel widths and attention head count while
 * leaving checkpoint-specific tensor naming and timestep/conditioning mapping
 * to the model-family adapter. The current block implementations use 32-group
 * normalization, so both feature widths must be divisible by 32.</p>
 */
public record UNet2DConditionalConfig(
        int inChannels,
        int outChannels,
        int tembDim,
        int contextDim,
        int baseChannels,
        int expandedChannels,
        int attentionHeads) {

    public static final int NORMALIZATION_GROUPS = 32;

    public UNet2DConditionalConfig {
        if (inChannels <= 0 || outChannels <= 0 || tembDim <= 0 || contextDim <= 0
                || baseChannels <= 0 || expandedChannels <= 0 || attentionHeads <= 0) {
            throw new IllegalArgumentException("UNet dimensions must be positive");
        }
        if (baseChannels % NORMALIZATION_GROUPS != 0
                || expandedChannels % NORMALIZATION_GROUPS != 0) {
            throw new IllegalArgumentException(
                    "baseChannels and expandedChannels must be divisible by " + NORMALIZATION_GROUPS);
        }
        if (baseChannels % attentionHeads != 0 || expandedChannels % attentionHeads != 0) {
            throw new IllegalArgumentException("channel widths must be divisible by attentionHeads");
        }
        Math.multiplyExact(tembDim, 4);
    }

    /** Standard scaffold dimensions retained for source compatibility. */
    public static UNet2DConditionalConfig standard(
            int inChannels, int outChannels, int tembDim, int contextDim) {
        return new UNet2DConditionalConfig(
                inChannels, outChannels, tembDim, contextDim, 320, 640, 8);
    }

    public int timeEmbeddingDim() {
        return Math.multiplyExact(tembDim, 4);
    }

    public int baseHeadDim() {
        return baseChannels / attentionHeads;
    }

    public int expandedHeadDim() {
        return expandedChannels / attentionHeads;
    }
}
