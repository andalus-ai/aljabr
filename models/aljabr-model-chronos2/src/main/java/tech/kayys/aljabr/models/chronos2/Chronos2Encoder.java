package tech.kayys.aljabr.models.chronos2;

import java.util.Objects;
import tech.kayys.aljabr.models.common.transformer.RmsNormReference;
import tech.kayys.aljabr.models.common.transformer.RotaryPositionEmbeddingReference;

/**
 * Float32 reference execution for the Chronos-2 encoder stack.
 *
 * <p>Input layout is [series][patch][modelDimension]. Each block applies
 * pre-normalized time attention, pre-normalized group attention, and a
 * pre-normalized ReLU feed-forward residual in that order. Time attention
 * uses RoPE; group attention deliberately does not because series have no
 * natural ordering. Attention scores are unscaled, matching the upstream
 * Chronos-2 eager implementation.</p>
 *
 * <p>This is a correctness reference, not an optimized backend kernel. All
 * parameters are supplied by the checkpoint mapper/loader.</p>
 */
public final class Chronos2Encoder {
    private final Chronos2Config config;
    private final LayerWeights[] layers;
    private final float[] finalNormScale;

    public Chronos2Encoder(Chronos2Config config, LayerWeights[] layers, float[] finalNormScale) {
        this.config = Objects.requireNonNull(config, "config");
        this.layers = Objects.requireNonNull(layers, "layers").clone();
        if (this.layers.length != config.layerCount()) {
            throw new IllegalArgumentException("layer weight count must equal configured layer count");
        }
        for (LayerWeights layer : this.layers) Objects.requireNonNull(layer, "layer weights");
        requireVector(finalNormScale, config.modelDimension(), "finalNormScale");
        this.finalNormScale = finalNormScale.clone();
    }

    /**
     * Encodes grouped time series.
     *
     * @param inputs [series][patch][modelDimension]
     * @param groupIds one group identifier per series
     * @param validPatches [series][patch], true for patches containing observations
     * @param positionIds one time position per patch (shared across series)
     * @return [series][patch][modelDimension]
     */
    public float[][][] encode(float[][][] inputs, int[] groupIds,
            boolean[][] validPatches, int[] positionIds) {
        Shape shape = validateInputs(inputs, groupIds, validPatches, positionIds);
        float[][][] hidden = copy(inputs);
        for (LayerWeights layer : layers) {
            // Time self-attention: independent temporal sequence per series.
            for (int series = 0; series < shape.series; series++) {
                float[][] normalized = normalize(hidden[series], layer.timeAttention().normScale());
                float[][] attended = attention(normalized, validPatches[series], positionIds,
                        layer.timeAttention().projections(), true);
                addInPlace(hidden[series], attended);
            }

            // Group self-attention: independent series set at each patch index.
            for (int patch = 0; patch < shape.patches; patch++) {
                float[][] slice = new float[shape.series][shape.dimension];
                boolean[] validKeys = new boolean[shape.series];
                for (int series = 0; series < shape.series; series++) {
                    slice[series] = hidden[series][patch].clone();
                    validKeys[series] = validPatches[series][patch];
                }
                float[][] normalized = normalize(slice, layer.groupAttention().normScale());
                float[][] attended = attention(normalized, validKeys, null,
                        layer.groupAttention().projections(), false, groupIds);
                for (int series = 0; series < shape.series; series++) {
                    addInPlace(hidden[series][patch], attended[series]);
                }
            }

            // Feed-forward residual, shared position-wise across series/patches.
            for (int series = 0; series < shape.series; series++) {
                for (int patch = 0; patch < shape.patches; patch++) {
                    float[] residual = hidden[series][patch];
                    float[] normalized = normalize(residual, layer.feedForward().normScale());
                    float[] projected = linear(normalized, layer.feedForward().wi());
                    for (int i = 0; i < projected.length; i++) projected[i] = Math.max(0f, projected[i]);
                    addInPlace(residual, linear(projected, layer.feedForward().wo()));
                }
            }
        }

        for (int series = 0; series < shape.series; series++) {
            for (int patch = 0; patch < shape.patches; patch++) {
                hidden[series][patch] = normalize(hidden[series][patch], finalNormScale);
            }
        }
        return hidden;
    }

    private float[][] attention(float[][] input, boolean[] validKeys, int[] positions,
            AttentionWeights weights, boolean useRope) {
        return attention(input, validKeys, positions, weights, useRope, null);
    }

    private float[][] attention(float[][] input, boolean[] validKeys, int[] positions,
            AttentionWeights weights, boolean useRope, int[] groupIds) {
        int sequence = input.length;
        int model = config.modelDimension();
        int heads = config.headCount();
        int headDim = config.keyValueDimension();
        int inner = Math.multiplyExact(heads, headDim);
        float[][] qFlat = linear(input, weights.query());
        float[][] kFlat = linear(input, weights.key());
        float[][] vFlat = linear(input, weights.value());
        float[][][] q = splitHeads(qFlat, heads, headDim);
        float[][][] k = splitHeads(kFlat, heads, headDim);
        float[][][] v = splitHeads(vFlat, heads, headDim);
        if (useRope) {
            q = RotaryPositionEmbeddingReference.apply(q, positions, config.ropeTheta(), null);
            k = RotaryPositionEmbeddingReference.apply(k, positions, config.ropeTheta(), null);
        }

        float[][] merged = new float[sequence][inner];
        for (int head = 0; head < heads; head++) {
            for (int query = 0; query < sequence; query++) {
                double max = Double.NEGATIVE_INFINITY;
                double[] scores = new double[sequence];
                for (int key = 0; key < sequence; key++) {
                    boolean visible = validKeys[key]
                            && (groupIds == null || groupIds[query] == groupIds[key]);
                    if (!visible) {
                        scores[key] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double dot = 0d;
                    for (int d = 0; d < headDim; d++) dot += (double) q[query][head][d] * k[key][head][d];
                    scores[key] = dot; // Chronos-2 intentionally uses scale=1.0.
                    max = Math.max(max, dot);
                }
                if (max == Double.NEGATIVE_INFINITY) continue; // Fully masked query: zero attention update.
                double denominator = 0d;
                for (double score : scores) if (score != Double.NEGATIVE_INFINITY) denominator += Math.exp(score - max);
                for (int key = 0; key < sequence; key++) {
                    if (scores[key] == Double.NEGATIVE_INFINITY) continue;
                    double probability = Math.exp(scores[key] - max) / denominator;
                    for (int d = 0; d < headDim; d++) {
                        merged[query][head * headDim + d] += (float) (probability * v[key][head][d]);
                    }
                }
            }
        }
        return linear(merged, weights.output());
    }

    private float[][] normalize(float[][] input, float[] scale) {
        float[][][] expanded = new float[input.length][1][];
        for (int i = 0; i < input.length; i++) expanded[i][0] = input[i];
        float[][][] result = RmsNormReference.apply(expanded, scale, (float) config.layerNormEpsilon());
        float[][] output = new float[input.length][];
        for (int i = 0; i < input.length; i++) output[i] = result[i][0];
        return output;
    }

    private float[] normalize(float[] input, float[] scale) {
        return normalize(new float[][] {input}, scale)[0];
    }

    private static float[][][] splitHeads(float[][] input, int heads, int headDim) {
        if (input.length == 0 || input[0].length != heads * headDim) {
            throw new IllegalArgumentException("projection output width must equal numHeads * dKv");
        }
        float[][][] output = new float[input.length][heads][headDim];
        for (int token = 0; token < input.length; token++) {
            for (int head = 0; head < heads; head++) {
                System.arraycopy(input[token], head * headDim, output[token][head], 0, headDim);
            }
        }
        return output;
    }

    /** Matrix layout is [output][input], matching PyTorch Linear checkpoints. */
    private static float[][] linear(float[][] input, float[][] weights) {
        if (input.length == 0 || weights == null || weights.length == 0
                || weights[0] == null || weights[0].length != input[0].length) {
            throw new IllegalArgumentException("linear weight input dimension mismatch");
        }
        float[][] output = new float[input.length][weights.length];
        for (int row = 0; row < input.length; row++) {
            if (input[row].length != weights[0].length) throw new IllegalArgumentException("input rows must be rectangular");
            for (int out = 0; out < weights.length; out++) {
                if (weights[out].length != input[row].length) throw new IllegalArgumentException("weight rows must be rectangular");
                double sum = 0d;
                for (int in = 0; in < input[row].length; in++) sum += (double) weights[out][in] * input[row][in];
                output[row][out] = (float) sum;
            }
        }
        return output;
    }

    private static float[] linear(float[] input, float[][] weights) {
        return linear(new float[][] {input}, weights)[0];
    }

    private static void addInPlace(float[] target, float[] update) {
        if (target.length != update.length) throw new IllegalArgumentException("residual dimensions must match");
        for (int i = 0; i < target.length; i++) target[i] += update[i];
    }

    private static void addInPlace(float[][] target, float[][] update) {
        if (target.length != update.length) throw new IllegalArgumentException("residual sequence lengths must match");
        for (int i = 0; i < target.length; i++) addInPlace(target[i], update[i]);
    }

    private static float[][][] copy(float[][][] input) {
        float[][][] result = new float[input.length][][];
        for (int i = 0; i < input.length; i++) {
            result[i] = new float[input[i].length][];
            for (int j = 0; j < input[i].length; j++) result[i][j] = input[i][j].clone();
        }
        return result;
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) throw new IllegalArgumentException(name + " must have length " + length);
        for (float value : vector) if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must be finite");
    }

    private Shape validateInputs(float[][][] inputs, int[] groupIds,
            boolean[][] valid, int[] positions) {
        if (inputs == null || inputs.length == 0 || groupIds == null || groupIds.length != inputs.length
                || valid == null || valid.length != inputs.length) {
            throw new IllegalArgumentException("series, groupIds, and validity masks must have equal non-zero lengths");
        }
        if (inputs[0] == null || inputs[0].length == 0) throw new IllegalArgumentException("each series must contain patches");
        int patches = inputs[0].length;
        int dimension = config.modelDimension();
        if (positions == null || positions.length != patches) throw new IllegalArgumentException("positionIds must match patch count");
        for (int s = 0; s < inputs.length; s++) {
            if (inputs[s] == null || inputs[s].length != patches || valid[s] == null || valid[s].length != patches) {
                throw new IllegalArgumentException("all series must have equal patch and mask lengths");
            }
            for (float[] patch : inputs[s]) {
                if (patch == null || patch.length != dimension) throw new IllegalArgumentException("patch width must equal modelDimension");
                for (float value : patch) if (!Float.isFinite(value)) throw new IllegalArgumentException("inputs must be finite");
            }
        }
        return new Shape(inputs.length, patches, dimension);
    }

    private record Shape(int series, int patches, int dimension) { }

    public record AttentionWeights(float[][] query, float[][] key, float[][] value, float[][] output) {
        public AttentionWeights {
            Objects.requireNonNull(query, "query");
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            Objects.requireNonNull(output, "output");
        }
    }

    public record AttentionLayerWeights(float[] normScale, AttentionWeights projections) {
        public AttentionLayerWeights {
            Objects.requireNonNull(normScale, "normScale");
            Objects.requireNonNull(projections, "projections");
        }
    }

    public record FeedForwardWeights(float[] normScale, float[][] wi, float[][] wo) {
        public FeedForwardWeights {
            Objects.requireNonNull(normScale, "normScale");
            Objects.requireNonNull(wi, "wi");
            Objects.requireNonNull(wo, "wo");
        }
    }

    public record LayerWeights(AttentionLayerWeights timeAttention,
            AttentionLayerWeights groupAttention, FeedForwardWeights feedForward) {
        public LayerWeights {
            Objects.requireNonNull(timeAttention, "timeAttention");
            Objects.requireNonNull(groupAttention, "groupAttention");
            Objects.requireNonNull(feedForward, "feedForward");
        }
    }
}
