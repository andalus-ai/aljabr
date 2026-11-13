package tech.kayys.aljabr.core.tensor.lazy;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class LazyTensorAutogradTest {

    @Test
    void reversePassAccumulatesFanOutIntoDifferentiableLeaf() {
        CpuBackend backend = new CpuBackend();
        Tensor leafValue = TensorFactory.of(new float[] { 3.0f }, backend, 1);
        leafValue.setRequiresGrad(true);
        LazyTensor leaf = LazyTensor.of(leafValue);
        Tensor output = leaf.mul(leaf).add(leaf);

        try {
            assertEquals(12.0f, output.item(), 1.0e-6f);
            output.backward();

            // d(x*x + x)/dx = 2*x + 1; both edges to the same leaf must accumulate.
            assertEquals(7.0f, leaf.grad().item(), 1.0e-6f);
        } finally {
            output.release();
            leafValue.release();
            backend.close();
        }
    }

    @Test
    void backwardRejectsOperationsWithoutRegisteredVjp() {
        CpuBackend backend = new CpuBackend();
        Tensor leafValue = TensorFactory.of(new float[] { 2.0f }, backend, 1);
        leafValue.setRequiresGrad(true);
        LazyTensor leaf = new LazyTensor(GraphNode.constant(leafValue), backend);
        Tensor output = leaf.relu();

        try {
            UnsupportedOperationException error =
                    assertThrows(UnsupportedOperationException.class, output::backward);
            assertEquals(true, error.getMessage().contains("No autograd rule registered for operation RELU"));
        } finally {
            output.release();
            leafValue.release();
            backend.close();
        }
    }

    @Test
    void implicitBackwardRequiresScalarOutput() {
        CpuBackend backend = new CpuBackend();
        Tensor leafValue = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2);
        leafValue.setRequiresGrad(true);
        LazyTensor leaf = new LazyTensor(GraphNode.constant(leafValue), backend);
        Tensor output = leaf.mul(2.0f);

        try {
            assertThrows(UnsupportedOperationException.class, output::backward);
        } finally {
            output.release();
            leafValue.release();
            backend.close();
        }
    }
    @Test
    void explicitOutputGradientSupportsNonScalarResults() {
        CpuBackend backend = new CpuBackend();
        Tensor leafValue = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2);
        leafValue.setRequiresGrad(true);
        LazyTensor leaf = LazyTensor.of(leafValue);
        Tensor output = leaf.mul(2.0f);
        Tensor seed = TensorFactory.of(new float[] { 3.0f, 4.0f }, backend, 2);

        try {
            output.backward(seed);
            assertArrayEquals(new float[] { 6.0f, 8.0f }, leaf.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            output.release();
            leafValue.release();
            backend.close();
        }
    }

}
