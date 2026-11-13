package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;

class DefaultTensorAutogradTest {

    @Test
    void eagerArithmeticRecordsGraphAndAccumulatesSharedLeafGradients() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 3.0f }, backend, 1);
        x.setRequiresGrad(true);
        Tensor square = x.mul(x);
        Tensor output = square.add(x);

        try {
            assertEquals(12.0f, output.item(), 1.0e-6f);
            assertEquals(true, output.requiresGrad());

            output.backward();

            // d(x*x + x)/dx = 2*x + 1.
            assertEquals(7.0f, x.grad().item(), 1.0e-6f);
        } finally {
            output.release();
            square.release();
            if (x.grad() != null) {
                x.grad().release();
            }
            x.release();
            backend.close();
        }
    }

    @Test
    void eagerScalarArithmeticSupportsExplicitVectorGradient() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2);
        x.setRequiresGrad(true);
        Tensor output = x.mul(2.0f);
        Tensor seed = TensorFactory.of(new float[] { 3.0f, 4.0f }, backend, 2);

        try {
            output.backward(seed);
            assertArrayEquals(new float[] { 6.0f, 8.0f }, x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            output.release();
            if (x.grad() != null) {
                x.grad().release();
            }
            x.release();
            backend.close();
        }
    }

    @Test
    void eagerNonScalarBackwardRequiresExplicitSeed() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2);
        x.setRequiresGrad(true);
        Tensor output = x.mul(2.0f);

        try {
            assertThrows(UnsupportedOperationException.class, output::backward);
        } finally {
            output.release();
            x.release();
            backend.close();
        }
    }
}
    @Test
    void eagerMeanPropagatesGradientAcrossAllElements() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2);
        x.setRequiresGrad(true);
        Tensor square = x.mul(x);
        Tensor loss = square.mean();

        try {
            assertEquals(2.5f, loss.item(), 1.0e-6f);
            loss.backward();
            assertArrayEquals(new float[] { 1.0f, 2.0f }, x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            loss.release();
            square.release();
            if (x.grad() != null) {
                x.grad().release();
            }
            x.release();
            backend.close();
        }
    }

