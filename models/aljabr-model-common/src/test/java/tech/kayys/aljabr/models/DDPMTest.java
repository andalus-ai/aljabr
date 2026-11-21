package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class DDPMTest {

    @Test
    void samplesConfiguredVectorDimensionOnSelectedBackend() {
        ComputeBackend backend = new CpuBackend();
        try (DDPM model = new DDPM(backend, 6, 12, 4);
             Tensor sample = model.sample(3, 1234L)) {
            assertEquals(backend, model.computeBackend());
            assertEquals(backend, sample.backend());
            assertEquals(backend, model.children().get("denoiser").computeBackend());
            model.children().get("denoiser").stateDict().values()
                    .forEach(tensor -> assertEquals(backend, tensor.backend()));
            assertEquals(2, sample.shape().rank());
            assertEquals(3, sample.shape().dim(0));
            assertEquals(6, sample.shape().dim(1));
            for (float value : sample.toFloatArray()) {
                assertTrue(Float.isFinite(value));
            }
        }
    }

    @Test
    void samplingIsDeterministicForSameModelAndSeed() {
        ComputeBackend backend = new CpuBackend();
        try (DDPM model = new DDPM(backend, 4, 8, 5);
             Tensor first = model.sample(2, 99L);
             Tensor second = model.sample(2, 99L)) {
            assertArrayEquals(first.toFloatArray(), second.toFloatArray(), 0f);
        }
    }

    @Test
    void differentSeedsChangeInitialAndReverseNoise() {
        try (DDPM model = new DDPM(new CpuBackend(), 4, 8, 5);
             Tensor first = model.sample(2, 11L);
             Tensor second = model.sample(2, 12L)) {
            assertFalse(java.util.Arrays.equals(first.toFloatArray(), second.toFloatArray()));
        }
    }

    @Test
    void rejectsInvalidConstructorDimensionsAndSchedule() {
        ComputeBackend backend = new CpuBackend();
        assertThrows(IllegalArgumentException.class, () -> new DDPM(backend, 0, 8, 4));
        assertThrows(IllegalArgumentException.class, () -> new DDPM(backend, 4, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> new DDPM(backend, 4, 8, 1));
        assertThrows(IllegalArgumentException.class, () -> new DDPM(backend, 4, 8, 0));
        assertThrows(NullPointerException.class, () -> new DDPM(null, 4, 8, 4));
    }

    @Test
    void rejectsInvalidSampleBatchSize() {
        try (DDPM model = new DDPM(new CpuBackend(), 4, 8, 3)) {
            assertThrows(IllegalArgumentException.class, () -> model.sample(0, 1L));
            assertThrows(IllegalArgumentException.class, () -> model.sample(-1, 1L));
        }
    }

    @Test
    void rejectsTrainingInputWithWrongShape() {
        ComputeBackend backend = new CpuBackend();
        try (DDPM model = new DDPM(backend, 4, 8, 3);
             Tensor wrongWidth = TensorFactory.of(new float[6], backend, 2, 3)) {
            assertThrows(IllegalArgumentException.class, () -> model.trainingLoss(wrongWidth));
        }
    }

    @Test
    void rejectsTrainingInputOwnedByAnotherBackend() {
        ComputeBackend modelBackend = new CpuBackend();
        ComputeBackend inputBackend = new CpuBackend();
        try (DDPM model = new DDPM(modelBackend, 4, 8, 3);
             Tensor input = TensorFactory.of(new float[8], inputBackend, 2, 4)) {
            assertThrows(IllegalArgumentException.class, () -> model.trainingLoss(input));
        }
    }

    @Test
    void trainingLossReturnsScalarOnSelectedBackend() {
        ComputeBackend backend = new CpuBackend();
        try (DDPM model = new DDPM(backend, 4, 8, 3);
             Tensor input = TensorFactory.of(new float[8], backend, 2, 4);
             Tensor loss = model.trainingLoss(input)) {
            assertEquals(backend, loss.backend());
            assertEquals(1L, loss.shape().numel());
        }
    }
}
