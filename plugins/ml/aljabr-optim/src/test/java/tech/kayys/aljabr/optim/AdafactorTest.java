package tech.kayys.aljabr.optim;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class AdafactorTest {

    @Test
    void validatesHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new Adafactor(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Adafactor(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Adafactor(0.1f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new Adafactor(0.1f, Float.POSITIVE_INFINITY));
    }

    @Test
    void factoredMatrixUpdatesMatchIndependentTwoStepReference() {
        double learningRate = 0.05;
        double epsilon = 1.0e-6;
        double[][] parameterValues = {{1.0, -2.0}, {0.5, 3.0}};
        double[][] previousRows = new double[2][];
        double[][] previousColumns = new double[2][];
        float[][] gradients = {
                {{0.5f, -1.0f}, {1.5f, 0.25f}},
                {{-0.25f, 0.75f}, {0.5f, -1.25f}}
        };

        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(flatten(parameterValues), backend, 2, 2);
            try (Linear layer = new Linear(initial, null);
                 Adafactor optimizer = new Adafactor((float) learningRate, (float) epsilon)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                for (int step = 1; step <= gradients.length; step++) {
                    double beta2 = 1.0 - Math.pow(step, -0.8);
                    double[] rowObservation = new double[2];
                    double[] columnObservation = new double[2];
                    for (int row = 0; row < 2; row++) {
                        for (int column = 0; column < 2; column++) {
                            double square = gradients[step - 1][row][column]
                                    * gradients[step - 1][row][column];
                            rowObservation[row] += square / 2.0;
                            columnObservation[column] += square / 2.0;
                        }
                    }
                    if (previousRows[0] == null) {
                        previousRows[0] = new double[2];
                        previousColumns[0] = new double[2];
                    }
                    double rowAverageMean = 0.0;
                    for (int row = 0; row < 2; row++) {
                        previousRows[0][row] = beta2 * previousRows[0][row]
                                + (1.0 - beta2) * rowObservation[row];
                        rowAverageMean += previousRows[0][row] / 2.0;
                    }
                    for (int column = 0; column < 2; column++) {
                        previousColumns[0][column] = beta2 * previousColumns[0][column]
                                + (1.0 - beta2) * columnObservation[column];
                    }
                    for (int row = 0; row < 2; row++) {
                        for (int column = 0; column < 2; column++) {
                            double variance = previousRows[0][row] * previousColumns[0][column]
                                    / rowAverageMean;
                            double update = gradients[step - 1][row][column]
                                    / (Math.sqrt(variance) + epsilon);
                            parameterValues[row][column] -= learningRate * update;
                        }
                    }

                    Tensor gradient = TensorFactory.of(flatten(gradients[step - 1]), backend, 2, 2);
                    try {
                        optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    } finally {
                        gradient.release();
                    }
                    assertThat(parameter.tensor().toFloatArray())
                            .containsExactly(flatten(parameterValues));
                }
            }
        }
    }

    @Test
    void vectorParametersUseAnUnfactoredSecondMoment() {
        double learningRate = 0.1;
        double epsilon = 1.0e-6;
        double[] expected = {1.0, -1.0, 0.5};
        double[] secondMoment = new double[3];
        float[][] gradients = {{0.5f, -0.25f, 1.0f}, {-0.75f, 0.5f, 0.25f}};

        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] {1.0f, -1.0f, 0.5f}, backend, 3);
            try (Linear layer = new Linear(initial, null);
                 Adafactor optimizer = new Adafactor((float) learningRate, (float) epsilon)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                for (int step = 1; step <= gradients.length; step++) {
                    double beta2 = 1.0 - Math.pow(step, -0.8);
                    for (int index = 0; index < expected.length; index++) {
                        double gradientValue = gradients[step - 1][index];
                        secondMoment[index] = beta2 * secondMoment[index]
                                + (1.0 - beta2) * gradientValue * gradientValue;
                        expected[index] -= learningRate * gradientValue
                                / (Math.sqrt(secondMoment[index]) + epsilon);
                    }
                    Tensor gradient = TensorFactory.of(gradients[step - 1], backend, 3);
                    try {
                        optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    } finally {
                        gradient.release();
                    }
                    assertThat(parameter.tensor().toFloatArray())
                            .containsExactly(toFloats(expected));
                }
            }
        }
    }

    @Test
    void parameterReplacementRestartsFactoredStateAndStepSchedule() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 2, 2);
            Tensor replacement = TensorFactory.of(new float[] {2f, 3f, 4f, 5f}, backend, 2, 2);
            Tensor gradient = TensorFactory.of(new float[] {0.5f, 1f, -0.5f, 2f}, backend, 2, 2);
            try (Linear layer = new Linear(initial, null);
                 Adafactor optimizer = new Adafactor(0.1f, 1.0e-6f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    parameter.updateTensor(replacement);

                    double[] expected = {2, 3, 4, 5};
                    double[] g = {0.5, 1, -0.5, 2};
                    double[] row = {(g[0] * g[0] + g[1] * g[1]) / 2.0,
                            (g[2] * g[2] + g[3] * g[3]) / 2.0};
                    double[] column = {(g[0] * g[0] + g[2] * g[2]) / 2.0,
                            (g[1] * g[1] + g[3] * g[3]) / 2.0};
                    double rowMean = (row[0] + row[1]) / 2.0;
                    for (int r = 0; r < 2; r++) {
                        for (int c = 0; c < 2; c++) {
                            int index = r * 2 + c;
                            double variance = row[r] * column[c] / rowMean;
                            expected[index] -= 0.1 * g[index] / (Math.sqrt(variance) + 1.0e-6);
                        }
                    }

                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray()).containsExactly(toFloats(expected));
                } finally {
                    gradient.release();
                }
            }
        }
    }

    @Test
    void validatesAllGradientsBeforePublishingAnyUpdate() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Adafactor optimizer = new Adafactor(0.01f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            Tensor originalWeight = weight.tensor();
            Tensor validWeightGradient = TensorFactory.ones(backend, 2, 2);
            Tensor invalidBiasGradient = TensorFactory.zeros(backend, 3);
            try {
                assertThrows(IllegalArgumentException.class, () -> optimizer.step(
                        Map.of("weight", weight, "bias", bias),
                        Map.of("weight", validWeightGradient, "bias", invalidBiasGradient)));
                assertThat(weight.tensor()).isSameAs(originalWeight);
            } finally {
                validWeightGradient.release();
                invalidBiasGradient.release();
            }
        }
    }

    @Test
    void missingAndFrozenParametersAreSkipped() {
        try (CpuBackend backend = new CpuBackend();
             Linear layer = new Linear(backend, 2, 2, true);
             Adafactor optimizer = new Adafactor(0.01f)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            Parameter bias = layer.namedParameterObjects().get("bias");
            bias.setTrainable(false);
            Tensor gradient = TensorFactory.ones(backend, 2, 2);
            try {
                Tensor originalWeight = weight.tensor();
                Tensor originalBias = bias.tensor();
                optimizer.step(Map.of("weight", weight, "bias", bias), Map.of());
                assertThat(weight.tensor()).isSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);

                optimizer.step(Map.of("weight", weight, "bias", bias), Map.of("weight", gradient));
                assertThat(weight.tensor()).isNotSameAs(originalWeight);
                assertThat(bias.tensor()).isSameAs(originalBias);
            } finally {
                gradient.release();
            }
        }
    }

    @Test
    void resetRestartsMomentStateAndPreservesHyperparameters() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor initial = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 2, 2);
            Tensor gradient = TensorFactory.of(new float[] {0.5f, 1f, -0.5f, 2f}, backend, 2, 2);
            try (Linear layer = new Linear(initial, null);
                 Adafactor optimizer = new Adafactor(0.1f, 1.0e-6f)) {
                Parameter parameter = layer.namedParameterObjects().get("weight");
                try {
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    optimizer.reset();
                    double[] expected = parameter.tensor().toFloatArray();
                    double[] g = {0.5, 1, -0.5, 2};
                    double[] row = {(g[0] * g[0] + g[1] * g[1]) / 2.0,
                            (g[2] * g[2] + g[3] * g[3]) / 2.0};
                    double[] column = {(g[0] * g[0] + g[2] * g[2]) / 2.0,
                            (g[1] * g[1] + g[3] * g[3]) / 2.0};
                    double rowMean = (row[0] + row[1]) / 2.0;
                    for (int r = 0; r < 2; r++) {
                        for (int c = 0; c < 2; c++) {
                            int index = r * 2 + c;
                            expected[index] -= 0.1 * g[index]
                                    / (Math.sqrt(row[r] * column[c] / rowMean) + 1.0e-6);
                        }
                    }
                    optimizer.step(Map.of("weight", parameter), Map.of("weight", gradient));
                    assertThat(parameter.tensor().toFloatArray()).containsExactly(toFloats(expected));
                    assertThat(optimizer.learningRate()).isEqualTo(0.1f);
                    assertThat(optimizer.epsilon()).isEqualTo(1.0e-6f);
                } finally {
                    gradient.release();
                }
            }
        }
    }

    private static float[] flatten(double[][] values) {
        return new float[] {
                (float) values[0][0], (float) values[0][1],
                (float) values[1][0], (float) values[1][1]
        };
    }

    private static float[] flatten(float[][] values) {
        return new float[] {values[0][0], values[0][1], values[1][0], values[1][1]};
    }

    private static float[] toFloats(double[] values) {
        float[] result = new float[values.length];
        for (int index = 0; index < values.length; index++) result[index] = (float) values[index];
        return result;
    }
}
