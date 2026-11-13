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

    @Test
    void eagerSigmoidAndTanhOperationsParticipateInTheSameAutogradGraph() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 0.0f, 1.0f }, backend, 2);
        x.setRequiresGrad(true);
        Tensor sigmoid = x.sigmoid();
        Tensor loss = sigmoid.mean();

        try {
            loss.backward();
            float sigmoidAtZero = 0.5f;
            float sigmoidAtOne = (float) (1.0 / (1.0 + Math.exp(-1.0)));
            assertArrayEquals(
                    new float[] {
                            sigmoidAtZero * (1.0f - sigmoidAtZero) / 2.0f,
                            sigmoidAtOne * (1.0f - sigmoidAtOne) / 2.0f
                    },
                    x.grad().toFloatArray(),
                    1.0e-6f);
        } finally {
            loss.release();
            sigmoid.release();
            x.release();
            backend.close();
        }
    }


    @Test
    void eagerDimensionWiseMeanRestoresReducedAxisDuringBackward() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, backend, 2, 3);
        x.setRequiresGrad(true);
        Tensor rowMeans = x.mean(1, false);
        Tensor seed = TensorFactory.of(new float[] { 1, 4 }, backend, 2);
        try {
            rowMeans.backward(seed);
            assertArrayEquals(
                    new float[] { 1f / 3f, 1f / 3f, 1f / 3f, 4f / 3f, 4f / 3f, 4f / 3f },
                    x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            rowMeans.release();
            x.release();
            backend.close();
        }
    }

    @Test
    void eagerBroadcastAdditionReducesGradientToEachInputShape() {
        CpuBackend backend = new CpuBackend();
        Tensor matrix = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, backend, 2, 3);
        Tensor bias = TensorFactory.of(new float[] { 10, 20, 30 }, backend, 3);
        matrix.setRequiresGrad(true);
        bias.setRequiresGrad(true);
        Tensor sum = matrix.add(bias);
        Tensor loss = sum.sum();
        try {
            loss.backward();
            assertArrayEquals(new float[] { 1, 1, 1, 1, 1, 1 },
                    matrix.grad().toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] { 2, 2, 2 },
                    bias.grad().toFloatArray(), 1.0e-6f);
        } finally {
            loss.release();
            sum.release();
            matrix.release();
            bias.release();
            backend.close();
        }
    }


    @Test
    void eagerDimensionWiseSumSupportsKeepDimensionBackward() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, backend, 2, 3);
        x.setRequiresGrad(true);
        Tensor columnSums = x.sum(0, true);
        Tensor seed = TensorFactory.of(new float[] { 2, 3, 4 }, backend, 1, 3);
        try {
            columnSums.backward(seed);
            assertArrayEquals(new float[] { 2, 3, 4, 2, 3, 4 },
                    x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            columnSums.release();
            x.release();
            backend.close();
        }
    }


    @Test
    void eagerTransposeBackwardAppliesTheInverseDimensionSwap() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, backend, 2, 3);
        x.setRequiresGrad(true);
        Tensor transposed = x.transpose(0, 1);
        Tensor seed = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, backend, 3, 2);
        try {
            transposed.backward(seed);
            assertArrayEquals(new float[] { 1, 3, 5, 2, 4, 6 },
                    x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            transposed.release();
            x.release();
            backend.close();
        }
    }

    @Test
    void eagerReshapeFlattenUnsqueezeAndSqueezeRestoreOriginalShapeInBackward() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, backend, 2, 3);
        x.setRequiresGrad(true);
        Tensor reshaped = x.reshape(1, 2, 3);
        Tensor flattened = reshaped.flatten();
        Tensor squeezed = flattened.reshape(1, 6).squeeze();
        Tensor expanded = squeezed.unsqueeze(0);
        Tensor seed = TensorFactory.of(new float[] { 2, 3, 4, 5, 6, 7 }, backend, 1, 6);
        try {
            expanded.backward(seed);
            assertArrayEquals(new float[] { 2, 3, 4, 5, 6, 7 },
                    x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            expanded.release();
            squeezed.release();
            flattened.release();
            reshaped.release();
            x.release();
            backend.close();
        }
    }

    @Test
    void eagerSliceBackwardScattersGradientIntoTheOriginalCoordinates() {
        CpuBackend backend = new CpuBackend();
        Tensor x = TensorFactory.of(
                new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, backend, 3, 4);
        x.setRequiresGrad(true);
        Tensor sliced = x.slice(new long[] { 1, 1 }, new long[] { 2, 2 });
        Tensor seed = TensorFactory.of(new float[] { 10, 20, 30, 40 }, backend, 2, 2);
        try {
            sliced.backward(seed);
            assertArrayEquals(
                    new float[] { 0, 0, 0, 0, 0, 10, 20, 0, 0, 30, 40, 0 },
                    x.grad().toFloatArray(), 1.0e-6f);
        } finally {
            seed.release();
            sliced.release();
            x.release();
            backend.close();
        }
    }

}
