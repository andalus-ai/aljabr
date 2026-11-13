package tech.kayys.aljabr.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
    void gradientIsAvailableOnlyAfterBackward() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 2);
        try (AutogradContext context = new AutogradContext()) {
            Variable x = context.variable(input);
            assertEquals(null, x.grad());
        } finally {
            input.release();
            backend.close();
        }
    }
}
