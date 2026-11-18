package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CpuRotaryEmbeddingTest {

    @Test
    void appliesInterleavedRotaryPairsAtPositionOffset() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 1, 1, 1, 4);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 1, 10_000f, false);
            double angle0 = 1.0;
            double angle1 = 0.01;
            assertArrayEquals(new float[] {
                    (float) (Math.cos(angle0) - 2 * Math.sin(angle0)),
                    (float) (Math.sin(angle0) + 2 * Math.cos(angle0)),
                    (float) (3 * Math.cos(angle1) - 4 * Math.sin(angle1)),
                    (float) (3 * Math.sin(angle1) + 4 * Math.cos(angle1))
            }, output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void appliesNeoXHalfSplitRotaryPairing() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 1, 1, 1, 4);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 1, 10_000f, true);
            double firstPairAngle = 1.0;
            double secondPairAngle = 0.01;
            assertArrayEquals(new float[] {
                    (float) (Math.cos(firstPairAngle) - 3 * Math.sin(firstPairAngle)),
                    (float) (2 * Math.cos(secondPairAngle) - 4 * Math.sin(secondPairAngle)),
                    (float) (Math.sin(firstPairAngle) + 3 * Math.cos(firstPairAngle)),
                    (float) (2 * Math.sin(secondPairAngle) + 4 * Math.cos(secondPairAngle))
            }, output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void rejectsOddHeadDimension() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f}, backend, 1, 1, 1, 3);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.applyRoPE(input, 0, 10_000f, false));
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void rejectsInvalidFrequencyBase() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f}, backend, 1, 1, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.applyRoPE(input, 0, 1f, false));
        } finally {
            input.release();
            backend.close();
        }
    }
    @Test
    void configuredRoPERotatesOnlyPrefixAndPreservesTrailingDimensions() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 1, 1, 1, 4);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 1, 10_000f, 2, false, 1f);
            assertArrayEquals(new float[] {
                    (float) (Math.cos(1.0) - 2 * Math.sin(1.0)),
                    (float) (Math.sin(1.0) + 2 * Math.cos(1.0)),
                    3f, 4f
            }, output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void configuredRoPEAppliesLinearPositionScaling() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f}, backend, 1, 1, 1, 2);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 4, 10_000f, 2, false, 2f);
            double angle = 2.0; // position 4 divided by scaling factor 2
            assertArrayEquals(new float[] {(float) Math.cos(angle), (float) Math.sin(angle)},
                    output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void explicitPositionIdsSupportDifferentPositionsPerBatch() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f, 1f, 0f}, backend, 2, 1, 1, 2);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 10_000f, 2, false, 1f, new long[] {0L, 1L});
            float[] values = output.toFloatArray();
            assertArrayEquals(new float[] {1f, 0f, (float) Math.cos(1.0), (float) Math.sin(1.0)},
                    values, 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void explicitPositionIdsMustMatchBatchAndSequence() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f, 1f, 0f}, backend, 2, 1, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.applyRoPE(input, 10_000f, 2, false, 1f, new long[] {0L}));
        } finally {
            input.release();
            backend.close();
        }
    }


    @Test
    void explicitPositionIdsSupportNeoXPairingAndPartialDimensions() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f},
                backend, 1, 1, 1, 6);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 10_000f, 4, true, 1f, new long[] {1L});
            double angle0 = 1.0;
            double angle1 = 0.01;
            assertArrayEquals(new float[] {
                    (float) (Math.cos(angle0) - 3 * Math.sin(angle0)),
                    (float) (2 * Math.cos(angle1) - 4 * Math.sin(angle1)),
                    (float) (Math.sin(angle0) + 3 * Math.cos(angle0)),
                    (float) (2 * Math.sin(angle1) + 4 * Math.cos(angle1)),
                    5f, 6f
            }, output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void explicitPositionIdsApplyLinearScaling() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f}, backend, 1, 1, 1, 2);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 10_000f, 2, false, 2f, new long[] {4L});
            assertArrayEquals(new float[] {(float) Math.cos(2.0), (float) Math.sin(2.0)},
                    output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void explicitPositionIdsRejectNegativePositions() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f}, backend, 1, 1, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.applyRoPE(input, 10_000f, 2, false, 1f, new long[] {-1L}));
        } finally {
            input.release();
            backend.close();
        }
    }

}
