package tech.kayys.aljabr.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class AutogradContextTest {

    @Test
    void accumulatesGradientsAcrossRepeatedUses() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 2f, 3f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            Variable loss = x.mul(x).sum();
            loss.backward();

            assertArrayEquals(new float[] { 4f, 6f }, x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void computesScalarArithmeticGradient() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 2f, 4f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            Variable loss = x.mul(3f).mean();
            loss.backward();

            assertArrayEquals(new float[] { 1.5f, 1.5f }, x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void computesBothInputGradientsForDivision() {
        CpuBackend backend = new CpuBackend();
        Tensor numerator = TensorFactory.of(new float[] { 6f, 8f }, backend, 2);
        Tensor denominator = TensorFactory.of(new float[] { 2f, 4f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(numerator);
            Variable y = context.variable(denominator);
            x.div(y).sum().backward();

            assertArrayEquals(new float[] { 0.5f, 0.25f }, x.grad().toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] { -1.5f, -0.5f }, y.grad().toFloatArray(), 1.0e-6f);
        } finally {
            numerator.release();
            denominator.release();
            backend.close();
        }
    }

    @Test
    void computesMatrixMultiplicationGradients() {
        CpuBackend backend = new CpuBackend();
        Tensor left = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f }, backend, 2, 2);
        Tensor right = TensorFactory.of(new float[] { 5f, 6f, 7f, 8f }, backend, 2, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable a = context.variable(left);
            Variable b = context.variable(right);
            Variable loss = a.matmul(b).sum();
            loss.backward();

            assertArrayEquals(new float[] { 11f, 15f, 11f, 15f }, a.grad().toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] { 4f, 4f, 6f, 6f }, b.grad().toFloatArray(), 1.0e-6f);
        } finally {
            left.release();
            right.release();
            backend.close();
        }
    }

    @Test
    void rejectsBroadcastingUntilItsGradientReductionIsImplemented() {
        CpuBackend backend = new CpuBackend();
        Tensor left = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
        Tensor right = TensorFactory.of(new float[] { 3f }, backend, 1);
        try (AutogradContext context = new AutogradContext()) {
            Variable a = context.variable(left);
            Variable b = context.variable(right);
            assertThrows(UnsupportedOperationException.class, () -> a.add(b));
        } finally {
            left.release();
            right.release();
            backend.close();
        }
    }

    @Test
    void rejectsNonScalarImplicitBackward() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            assertThrows(IllegalArgumentException.class, () -> x.backward());
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void invalidNonScalarBackwardDoesNotConsumeContext() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            Variable nonScalar = x.mul(2f);

            assertThrows(IllegalArgumentException.class, nonScalar::backward);

            x.sum().backward();
            assertArrayEquals(new float[] { 1f, 1f }, x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void meanSquareGradientMatchesFiniteDifferences() {
        CpuBackend backend = new CpuBackend();
        float[] values = { 1.25f, -0.75f };
        Tensor input = TensorFactory.of(values, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            x.mul(x).mean().backward();

            float epsilon = 1.0e-3f;
            for (int i = 0; i < values.length; i++) {
                float original = values[i];
                values[i] = original + epsilon;
                float upper = meanSquare(backend, values);
                values[i] = original - epsilon;
                float lower = meanSquare(backend, values);
                values[i] = original;

                float numerical = (upper - lower) / (2.0f * epsilon);
                assertEquals(numerical, x.grad().toFloatArray()[i], 2.0e-3f);
            }
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void transposeBackpropagatesThroughInversePermutation() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, backend, 2, 3);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            x.transpose(0, 1).mul(x.transpose(0, 1)).sum().backward();

            assertArrayEquals(new float[] { 2f, 4f, 6f, 8f, 10f, 12f },
                    x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            input.release();
            backend.close();
        }
    }

    private static float meanSquare(CpuBackend backend, float[] values) {
        Tensor input = TensorFactory.of(values, backend, values.length);
        Tensor squared = null;
        Tensor mean = null;
        try {
            squared = input.mul(input);
            mean = squared.mean();
            return mean.item();
        } finally {
            if (mean != null) mean.release();
            if (squared != null) squared.release();
            input.release();
        }
    }

    @Test
    void gradientIsAvailableOnlyAfterBackward() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            assertNull(x.grad());
        } finally {
            input.release();
            backend.close();
        }
    }
}
