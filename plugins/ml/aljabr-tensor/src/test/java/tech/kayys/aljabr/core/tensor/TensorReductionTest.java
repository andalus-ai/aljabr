package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;

class TensorReductionTest {
    @Test
    void sumSupportsPositiveAndNegativeAxesWithBothShapePolicies() {
        ComputeBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, backend, 2, 3);
        Tensor axisZero = input.sum(0, false);
        Tensor axisOne = input.sum(-1, true);
        try {
            assertArrayEquals(new long[] {3}, axisZero.shape().dims());
            assertArrayEquals(new float[] {5f, 7f, 9f}, axisZero.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new long[] {2, 1}, axisOne.shape().dims());
            assertArrayEquals(new float[] {6f, 15f}, axisOne.toFloatArray(), 1.0e-6f);
        } finally {
            axisZero.release();
            axisOne.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void meanSupportsPositiveAndNegativeAxesWithBothShapePolicies() {
        ComputeBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, backend, 2, 3);
        Tensor axisZero = input.mean(0, false);
        Tensor axisOne = input.mean(-1, true);
        try {
            assertArrayEquals(new long[] {3}, axisZero.shape().dims());
            assertArrayEquals(new float[] {2.5f, 3.5f, 4.5f}, axisZero.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new long[] {2, 1}, axisOne.shape().dims());
            assertArrayEquals(new float[] {2f, 5f}, axisOne.toFloatArray(), 1.0e-6f);
        } finally {
            axisZero.release();
            axisOne.release();
            input.release();
            backend.close();
        }
    }

    @Test
    void axisReductionRejectsOutOfRangeAxes() {
        ComputeBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, backend, 2, 3);
        try {
            assertThrows(IllegalArgumentException.class, () -> input.sum(2, false));
            assertThrows(IllegalArgumentException.class, () -> input.sum(-3, false));
            assertThrows(IllegalArgumentException.class, () -> input.mean(2, true));
            assertThrows(IllegalArgumentException.class, () -> input.mean(-3, true));
        } finally {
            input.release();
            backend.close();
        }
    }

    @Test
    void axisReductionRejectsScalarInputs() {
        ComputeBackend backend = new CpuBackend();
        Tensor scalar = TensorFactory.of(new float[] {7f}, backend, new long[0]);
        try {
            assertThrows(IllegalArgumentException.class, () -> scalar.sum(0, false));
            assertThrows(IllegalArgumentException.class, () -> scalar.mean(-1, false));
        } finally {
            scalar.release();
            backend.close();
        }
    }

    @Test
    void meanRejectsEmptyInput() {
        ComputeBackend backend = new CpuBackend();
        Tensor empty = TensorFactory.of(new float[0], backend, 0);
        try {
            assertThrows(IllegalArgumentException.class, empty::mean);
        } finally {
            empty.release();
            backend.close();
        }
    }
}
