package tech.kayys.aljabr.models.chronos2;

import java.util.Objects;
import tech.kayys.aljabr.models.common.transformer.ResidualProjectionReference;
import tech.kayys.aljabr.models.common.transformer.ResidualProjectionReference.Activation;
import tech.kayys.aljabr.models.common.transformer.ResidualProjectionReference.Weights;

/**
 * Chronos-2 input patch embedding and output quantile projection reference.
 *
 * <p>Inputs are arranged as [series][patch][feature]. The regression token,
 * when enabled, is appended after context patches and before future patches.
 * Output projection follows upstream reshape order [series][quantile][horizon].</p>
 */
public final class Chronos2EmbeddingExecution {
    private final Chronos2Config config;
    private final Weights inputEmbedding;
    private final Weights outputEmbedding;
    private final float[][] sharedTokenEmbeddings;

    public Chronos2EmbeddingExecution(Chronos2Config config, Weights inputEmbedding,
            Weights outputEmbedding, float[][] sharedTokenEmbeddings) {
        this.config = Objects.requireNonNull(config, "config");
        this.inputEmbedding = Objects.requireNonNull(inputEmbedding, "inputEmbedding");
        this.outputEmbedding = Objects.requireNonNull(outputEmbedding, "outputEmbedding");
        validateProjection(inputEmbedding, Math.multiplyExact(config.inputPatchSize(), 3),
                config.feedForwardDimension(), config.modelDimension(), "input embedding");
        validateProjection(outputEmbedding, config.modelDimension(), config.feedForwardDimension(),
                Math.multiplyExact(config.quantiles().size(), config.outputPatchSize()), "output embedding");
        this.sharedTokenEmbeddings = copy(Objects.requireNonNull(sharedTokenEmbeddings, "sharedTokenEmbeddings"));
        int expectedTokens = config.useRegressionToken() ? 2 : 1;
        if (this.sharedTokenEmbeddings.length != expectedTokens) {
            throw new IllegalArgumentException("shared token embedding count must be " + expectedTokens);
        }
        for (float[] token : this.sharedTokenEmbeddings) {
            if (token.length != config.modelDimension()) {
                throw new IllegalArgumentException("shared token embedding width must equal modelDimension");
            }
            requireFinite(token, "shared token embedding");
        }
    }

    /**
     * Projects context and future patches into model dimension and optionally
     * appends the learned [REG] embedding to each series' context.
     *
     * @return [series][context patches + optional regression token + future patches][model dimension]
     */
    public float[][][] embed(float[][][] contextFeatures, float[][][] futureFeatures) {
        Shape shape = validateFeatures(contextFeatures, futureFeatures);
        int extra = config.useRegressionToken() ? 1 : 0;
        float[][][] result = new float[shape.series][shape.context + extra + shape.future][config.modelDimension()];
        for (int series = 0; series < shape.series; series++) {
            float[][] context = ResidualProjectionReference.apply(contextFeatures[series],
                    inputEmbedding, Activation.RELU);
            float[][] future = ResidualProjectionReference.apply(futureFeatures[series],
                    inputEmbedding, Activation.RELU);
            for (int patch = 0; patch < shape.context; patch++) result[series][patch] = context[patch];
            if (extra == 1) result[series][shape.context] = sharedTokenEmbeddings[1].clone();
            for (int patch = 0; patch < shape.future; patch++) {
                result[series][shape.context + extra + patch] = future[patch];
            }
        }
        return result;
    }

    /**
     * Projects future encoder states into quantiles, returning
     * [series][quantile][horizon] in the same order as upstream Chronos-2.
     */
    public float[][][] projectQuantiles(float[][][] futureHiddenStates) {
        Objects.requireNonNull(futureHiddenStates, "futureHiddenStates");
        if (futureHiddenStates.length == 0) throw new IllegalArgumentException("future states must not be empty");
        int quantileCount = config.quantiles().size();
        int patchSize = config.outputPatchSize();
        int horizon = Math.multiplyExact(futureHiddenStates[0].length, patchSize);
        if (futureHiddenStates[0].length == 0
                || futureHiddenStates[0][0].length != config.modelDimension()) {
            throw new IllegalArgumentException("future hidden states must have shape [series][patch][modelDimension]");
        }
        float[][][] output = new float[futureHiddenStates.length][quantileCount][horizon];
        for (int series = 0; series < futureHiddenStates.length; series++) {
            if (futureHiddenStates[series] == null || futureHiddenStates[series].length * patchSize != horizon) {
                throw new IllegalArgumentException("all series must have equal future patch counts");
            }
            float[][] projected = ResidualProjectionReference.apply(
                    futureHiddenStates[series], outputEmbedding, Activation.RELU);
            for (int patch = 0; patch < projected.length; patch++) {
                if (projected[patch].length != quantileCount * patchSize) {
                    throw new IllegalArgumentException("output embedding width must equal quantiles * outputPatchSize");
                }
                for (int quantile = 0; quantile < quantileCount; quantile++) {
                    for (int offset = 0; offset < patchSize; offset++) {
                        output[series][quantile][patch * patchSize + offset] =
                                projected[patch][quantile * patchSize + offset];
                    }
                }
            }
        }
        return output;
    }

    private Shape validateFeatures(float[][][] context, float[][][] future) {
        Objects.requireNonNull(context, "contextFeatures");
        Objects.requireNonNull(future, "futureFeatures");
        if (context.length == 0 || context.length != future.length) {
            throw new IllegalArgumentException("context and future series counts must match and be non-zero");
        }
        int width = Math.multiplyExact(config.inputPatchSize(), 3);
        int contextPatches = context[0].length;
        int futurePatches = future[0].length;
        if (contextPatches == 0 || futurePatches == 0) {
            throw new IllegalArgumentException("context and future patch sequences must not be empty");
        }
        for (int series = 0; series < context.length; series++) {
            validateRows(context[series], contextPatches, width, "context");
            validateRows(future[series], futurePatches, width, "future");
        }
        return new Shape(context.length, contextPatches, futurePatches);
    }

    private static void validateProjection(Weights weights, int inputWidth, int hiddenWidth,
            int outputWidth, String name) {
        float[][] hidden = weights.hiddenWeight();
        float[][] output = weights.outputWeight();
        float[][] residual = weights.residualWeight();
        if (hidden.length != hiddenWidth || hidden.length == 0 || hidden[0].length != inputWidth
                || output.length != outputWidth || output.length == 0 || output[0].length != hiddenWidth
                || residual.length != outputWidth || residual.length == 0 || residual[0].length != inputWidth) {
            throw new IllegalArgumentException(name + " weights do not match configured dimensions");
        }
    }

    private static void validateRows(float[][] rows, int expected, int width, String name) {
        if (rows == null || rows.length != expected) throw new IllegalArgumentException(name + " patch count mismatch");
        for (float[] row : rows) {
            if (row == null || row.length != width) throw new IllegalArgumentException(name + " feature width mismatch");
            requireFinite(row, name + " features");
        }
    }

    private static float[][] copy(float[][] values) {
        float[][] result = new float[values.length][];
        for (int i = 0; i < values.length; i++) result[i] = Objects.requireNonNull(values[i], "token").clone();
        return result;
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must contain finite values");
        }
    }

    private record Shape(int series, int context, int future) { }
}
