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
            assertEquals(7.0f, x.grad().item(), 1.0e-6f);
        } finally {
            output.release();
            square.release();
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
            x.release();
            backend.close();
        }
    }

    @Test
    void eagerMatmulAndMeanPropagateGradientsToBothMatrices() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, backend, 2, 2);
        Tensor weight = TensorFactory.of(new float[] { 2.0f, 3.0f }, backend, 2, 1);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);
        Tensor product = input.matmul(weight);
        Tensor loss = product.mean();

        try {
            assertEquals(13.0f, loss.item(), 1.0e-6f);
            loss.backward();
            assertArrayEquals(new float[] { 1.0f, 1.5f, 1.0f, 1.5f },
                    input.grad().toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] { 2.0f, 3.0f },
                    weight.grad().toFloatArray(), 1.0e-6f);
        } finally {
            loss.release();
            product.release();
            input.release();
            weight.release();
            backend.close();
        }
    }
    @Test
    void eagerUnsupportedOperationDoesNotSilentlyDetachGradientGraph() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { -1.0f, 2.0f }, backend, 2);
        x.setRequiresGrad(true);
        Tensor activated = x.relu();
        Tensor loss = activated.mean();

        try {
            assertEquals(true, loss.requiresGrad());
            assertThrows(UnsupportedOperationException.class, loss::backward);
        } finally {
            loss.release();
            activated.release();
            x.release();
            backend.close();
        }
    }

}
