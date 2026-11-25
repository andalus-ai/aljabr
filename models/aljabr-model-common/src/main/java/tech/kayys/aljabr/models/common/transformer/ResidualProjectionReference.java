package tech.kayys.aljabr.models.common.transformer;

import java.util.Objects;

/**
 * Backend-neutral float32 reference for a two-layer residual projection block.
 *
 * <p>The block computes {@code output(hidden(activation(input))) + residual(input)}.
 * Matrices use PyTorch Linear layout {@code [output][input]}. Biases are
 * explicit so the primitive can represent both biased embedding projections
 * and bias-free transformer projections.</p>
 */
public final class ResidualProjectionReference {
    private ResidualProjectionReference() { }

    public static float[][] apply(float[][] input, Weights weights, Activation activation) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(activation, "activation");
        if (input.length == 0) throw new IllegalArgumentException("input must not be empty");
        int inputWidth = input[0].length;
        validateMatrix(weights.hiddenWeight(), weights.hiddenBias(), inputWidth, "hidden");
        int hiddenWidth = weights.hiddenWeight().length;
        int outputWidth = weights.outputWeight().length;
        validateMatrix(weights.outputWeight(), weights.outputBias(), hiddenWidth, "output");
        validateMatrix(weights.residualWeight(), weights.residualBias(), inputWidth, "residual");
        if (weights.residualWeight().length != outputWidth) {
            throw new IllegalArgumentException("residual and output projections must have equal output widths");
        }

        float[][] result = new float[input.length][outputWidth];
        for (int row = 0; row < input.length; row++) {
            if (input[row] == null || input[row].length != inputWidth) {
                throw new IllegalArgumentException("input rows must be non-empty and rectangular");
            }
            requireFinite(input[row], "input");
            float[] hidden = linear(input[row], weights.hiddenWeight(), weights.hiddenBias());
            for (int i = 0; i < hidden.length; i++) hidden[i] = activation.apply(hidden[i]);
            float[] projected = linear(hidden, weights.outputWeight(), weights.outputBias());
            float[] residual = linear(input[row], weights.residualWeight(), weights.residualBias());
            for (int i = 0; i < outputWidth; i++) result[row][i] = projected[i] + residual[i];
        }
        return result;
    }

    private static float[] linear(float[] input, float[][] weight, float[] bias) {
        float[] output = new float[weight.length];
        for (int out = 0; out < weight.length; out++) {
            double sum = bias == null ? 0d : bias[out];
            for (int in = 0; in < input.length; in++) sum += (double) weight[out][in] * input[in];
            output[out] = (float) sum;
        }
        return output;
    }

    private static void validateMatrix(float[][] weight, float[] bias, int inputWidth, String name) {
        Objects.requireNonNull(weight, name + "Weight");
        if (weight.length == 0) throw new IllegalArgumentException(name + " weight must not be empty");
        if (bias != null && bias.length != weight.length) {
            throw new IllegalArgumentException(name + " bias width must match output width");
        }
        if (bias != null) requireFinite(bias, name + " bias");
        for (float[] row : weight) {
            if (row == null || row.length != inputWidth) {
                throw new IllegalArgumentException(name + " weight must have input width " + inputWidth);
            }
            requireFinite(row, name + " weight");
        }
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must contain finite values");
        }
    }

    public enum Activation {
        IDENTITY { @Override float apply(float value) { return value; } },
        RELU { @Override float apply(float value) { return Math.max(0f, value); } },
        TANH { @Override float apply(float value) { return (float) Math.tanh(value); } };

        abstract float apply(float value);
    }

    /** Immutable logical weights; null bias denotes a bias-free linear layer. */
    public record Weights(float[][] hiddenWeight, float[] hiddenBias,
            float[][] outputWeight, float[] outputBias,
            float[][] residualWeight, float[] residualBias) {
        public Weights {
            hiddenWeight = copy(Objects.requireNonNull(hiddenWeight, "hiddenWeight"));
            hiddenBias = hiddenBias == null ? null : hiddenBias.clone();
            outputWeight = copy(Objects.requireNonNull(outputWeight, "outputWeight"));
            outputBias = outputBias == null ? null : outputBias.clone();
            residualWeight = copy(Objects.requireNonNull(residualWeight, "residualWeight"));
            residualBias = residualBias == null ? null : residualBias.clone();
        }
        @Override public float[][] hiddenWeight() { return copy(hiddenWeight); }
        @Override public float[] hiddenBias() { return hiddenBias == null ? null : hiddenBias.clone(); }
        @Override public float[][] outputWeight() { return copy(outputWeight); }
        @Override public float[] outputBias() { return outputBias == null ? null : outputBias.clone(); }
        @Override public float[][] residualWeight() { return copy(residualWeight); }
        @Override public float[] residualBias() { return residualBias == null ? null : residualBias.clone(); }

        private static float[][] copy(float[][] source) {
            float[][] result = new float[source.length][];
            for (int i = 0; i < source.length; i++) result[i] = Objects.requireNonNull(source[i], "weight row").clone();
            return result;
        }
    }
}
