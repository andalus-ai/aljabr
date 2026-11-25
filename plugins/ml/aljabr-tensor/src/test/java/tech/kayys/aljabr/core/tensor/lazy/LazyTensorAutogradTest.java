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


    @Test
    void indexSelectBackwardAccumulatesRepeatedIndexContributions() {
        CpuBackend backend = new CpuBackend();
        Tensor leafValue = TensorFactory.of(new float[] { 10.0f, 20.0f, 30.0f }, backend, 3);
        leafValue.setRequiresGrad(true);
        LazyTensor leaf = LazyTensor.of(leafValue);
        Tensor indices = TensorFactory.of(new int[] { 2, 0, 2 }, backend, 3);
        Tensor selected = leaf.indexSelect(0, indices);
        Tensor seed = TensorFactory.of(new float[] { 1.0f, 2.0f, 3.0f }, backend, 3);

        try {
            selected.backward(seed);
            assertArrayEquals(new float[] { 2.0f, 0.0f, 4.0f }, leaf.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            selected.release();
            indices.release();
            leafValue.release();
            backend.close();
        }
    }

    @Test
    void lazyIndexAddRejectsSourceWithDifferentDtype() {
        CpuBackend backend = new CpuBackend();
        Tensor destinationValue = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2);
        Tensor integerSource = TensorFactory.of(new int[] { 3 }, backend, 1);
        LazyTensor destination = LazyTensor.of(destinationValue);
        Tensor indices = TensorFactory.of(new int[] { 0 }, backend, 1);

        try {
            assertThrows(IllegalArgumentException.class,
                    () -> destination.indexAdd(0, indices, integerSource));
        } finally {
            indices.release();
            destination.release();
            destinationValue.release();
            integerSource.release();
            backend.close();
        }
    }

    @Test
    void indexAddBackwardRoutesGradientsToDestinationAndSource() {
        CpuBackend backend = new CpuBackend();
        Tensor destinationValue = TensorFactory.of(new float[] { 1.0f, 2.0f, 3.0f }, backend, 3);
        Tensor sourceValue = TensorFactory.of(new float[] { 4.0f, 5.0f }, backend, 2);
        destinationValue.setRequiresGrad(true);
        sourceValue.setRequiresGrad(true);
        LazyTensor destination = LazyTensor.of(destinationValue);
        LazyTensor source = LazyTensor.of(sourceValue);
        Tensor indices = TensorFactory.of(new int[] { 1, 1 }, backend, 2);
        Tensor output = destination.indexAdd(0, indices, source);
        Tensor seed = TensorFactory.of(new float[] { 1.0f, 2.0f, 3.0f }, backend, 3);

        try {
            assertArrayEquals(new float[] { 1.0f, 11.0f, 3.0f }, output.toFloatArray(), 1.0e-6f);
            output.backward(seed);
            assertArrayEquals(new float[] { 1.0f, 2.0f, 3.0f },
                    destination.grad().toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] { 2.0f, 2.0f },
                    source.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            output.release();
            indices.release();
            destinationValue.release();
            sourceValue.release();
            backend.close();
        }
    }
}
