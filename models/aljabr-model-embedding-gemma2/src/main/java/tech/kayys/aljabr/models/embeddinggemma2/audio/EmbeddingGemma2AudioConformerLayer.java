package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Checkpoint-backed Java reference for one Gemma 4 audio Conformer layer.
 *
 * <p>The operation order follows the published layer exactly:
 * feed-forward-1, pre-attention clamp and RMSNorm, chunked relative attention,
 * attention-output clamp and RMSNorm, attention residual addition, light
 * causal convolution, feed-forward-2, output clamp, and final RMSNorm.</p>
 *
 * <p>All parameters are supplied explicitly by a future checkpoint mapper.
 * This class performs no parameter initialization or loading and remains a
 * Java-float reference implementation; it does not replace Aljabr Tensor
 * operations or backend kernels.</p>
 */
public final class EmbeddingGemma2AudioConformerLayer {
    private final float gradientClipping;
    private final float epsilon;
    private final float[] preAttentionNormScale;
    private final float[] postAttentionNormScale;
    private final float[] outputNormScale;
    private final EmbeddingGemma2AudioFeedForward feedForward1;
    private final EmbeddingGemma2AudioAttention selfAttention;
    private final EmbeddingGemma2AudioLightConv1d lightConvolution;
    private final EmbeddingGemma2AudioFeedForward feedForward2;
    private final int hiddenSize;

    public EmbeddingGemma2AudioConformerLayer(
            EmbeddingGemma2AudioConfig config,
            Weights weights) {
        Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(weights, "weights must not be null");
        this.hiddenSize = config.hiddenSize();
        this.gradientClipping = (float) Math.min(config.gradientClipping(), Float.MAX_VALUE);
        this.epsilon = (float) config.rmsNormEps();
        this.preAttentionNormScale = copyAndValidateScale(
                weights.preAttentionNormScale(), hiddenSize, "preAttentionNormScale");
        this.postAttentionNormScale = copyAndValidateScale(
                weights.postAttentionNormScale(), hiddenSize, "postAttentionNormScale");
        this.outputNormScale = copyAndValidateScale(
                weights.outputNormScale(), hiddenSize, "outputNormScale");
        this.feedForward1 = new EmbeddingGemma2AudioFeedForward(config, weights.feedForward1());
        this.selfAttention = new EmbeddingGemma2AudioAttention(config, weights.selfAttention());
        this.lightConvolution = new EmbeddingGemma2AudioLightConv1d(config, weights.lightConvolution());
        this.feedForward2 = new EmbeddingGemma2AudioFeedForward(config, weights.feedForward2());
    }

    /**
     * Applies one layer to [batch][sequence][hidden] states.
     *
     * @param hiddenStates input states
     * @param validMask optional true=valid frame mask [batch][sequence]
     * @return final normalized states; caller input is not mutated
     */
    public float[][][] compute(float[][][] hiddenStates, boolean[][] validMask) {
        validateInput(hiddenStates, validMask);

        float[][][] hidden = feedForward1.compute(hiddenStates);

        float[][][] residual = copy(hidden);
        hidden = rmsNorm(clamp(hidden, gradientClipping), preAttentionNormScale, epsilon);

        hidden = selfAttention.compute(hidden, validMask).hiddenStates();
        hidden = rmsNorm(clamp(hidden, gradientClipping), postAttentionNormScale, epsilon);
        addInPlace(hidden, residual);

        hidden = lightConvolution.compute(hidden);
        hidden = feedForward2.compute(hidden);

        return rmsNorm(clamp(hidden, gradientClipping), outputNormScale, epsilon);
    }

    private void validateInput(float[][][] input, boolean[][] mask) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("input must contain at least one batch item");
        }
        int sequence = -1;
        for (int b = 0; b < input.length; b++) {
            float[][] batch = input[b];
            if (batch == null || batch.length == 0) {
                throw new IllegalArgumentException("each batch item must contain at least one frame");
            }
            if (sequence < 0) sequence = batch.length;
            if (batch.length != sequence) {
                throw new IllegalArgumentException("input batch must be rectangular");
            }
            for (float[] frame : batch) {
                if (frame == null || frame.length != hiddenSize) {
                    throw new IllegalArgumentException("each frame must have hidden size " + hiddenSize);
                }
                for (float value : frame) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("input must contain only finite values");
                    }
                }
            }
        }
        if (mask != null) {
            if (mask.length != input.length) {
                throw new IllegalArgumentException("validMask batch dimension must match input");
            }
            for (boolean[] row : mask) {
                if (row == null || row.length != sequence) {
                    throw new IllegalArgumentException("validMask sequence dimension must match input");
                }
            }
        }
    }

    private static float[] copyAndValidateScale(float[] scale, int size, String name) {
        Objects.requireNonNull(scale, name + " must not be null");
        if (scale.length != size) {
            throw new IllegalArgumentException(name + " length must be " + size);
        }
        float[] result = scale.clone();
        for (float value : result) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain finite values");
            }
        }
        return result;
    }

    private static float[][][] clamp(float[][][] input, float limit) {
        float[][][] output = copy(input);
        for (float[][] batch : output) {
            for (float[] frame : batch) {
                for (int h = 0; h < frame.length; h++) {
                    frame[h] = Math.max(-limit, Math.min(limit, frame[h]));
                }
            }
        }
        return output;
    }

    private static float[][][] rmsNorm(float[][][] input, float[] scale, float epsilon) {
        float[][][] output = new float[input.length][input[0].length][scale.length];
        for (int b = 0; b < input.length; b++) {
            for (int t = 0; t < input[b].length; t++) {
                double meanSquare = 0.0;
                for (float value : input[b][t]) meanSquare += (double) value * value;
                meanSquare /= scale.length;
                float inverseRms = (float) (1.0 / Math.sqrt(meanSquare + epsilon));
                for (int h = 0; h < scale.length; h++) {
                    output[b][t][h] = input[b][t][h] * inverseRms * scale[h];
                }
            }
        }
        return output;
    }

    private static void addInPlace(float[][][] target, float[][][] residual) {
        for (int b = 0; b < target.length; b++) {
            for (int t = 0; t < target[b].length; t++) {
                for (int h = 0; h < target[b][t].length; h++) {
                    target[b][t][h] += residual[b][t][h];
                }
            }
        }
    }

    private static float[][][] copy(float[][][] input) {
        float[][][] result = new float[input.length][][];
        for (int b = 0; b < input.length; b++) {
            result[b] = new float[input[b].length][];
            for (int t = 0; t < input[b].length; t++) result[b][t] = input[b][t].clone();
        }
        return result;
    }

    /** Parameters for one layer, all supplied by the checkpoint mapper. */
    public record Weights(
            EmbeddingGemma2AudioFeedForward.Weights feedForward1,
            float[] preAttentionNormScale,
            EmbeddingGemma2AudioAttention.Weights selfAttention,
            float[] postAttentionNormScale,
            EmbeddingGemma2AudioLightConv1d.Weights lightConvolution,
            EmbeddingGemma2AudioFeedForward.Weights feedForward2,
            float[] outputNormScale) {
        public Weights {
            Objects.requireNonNull(feedForward1, "feedForward1 must not be null");
            Objects.requireNonNull(selfAttention, "selfAttention must not be null");
            Objects.requireNonNull(lightConvolution, "lightConvolution must not be null");
            Objects.requireNonNull(feedForward2, "feedForward2 must not be null");
            preAttentionNormScale = Objects.requireNonNull(
                    preAttentionNormScale, "preAttentionNormScale must not be null").clone();
            postAttentionNormScale = Objects.requireNonNull(
                    postAttentionNormScale, "postAttentionNormScale must not be null").clone();
            outputNormScale = Objects.requireNonNull(
                    outputNormScale, "outputNormScale must not be null").clone();
        }
        @Override public float[] preAttentionNormScale() { return preAttentionNormScale.clone(); }
        @Override public float[] postAttentionNormScale() { return postAttentionNormScale.clone(); }
        @Override public float[] outputNormScale() { return outputNormScale.clone(); }
    }
}
