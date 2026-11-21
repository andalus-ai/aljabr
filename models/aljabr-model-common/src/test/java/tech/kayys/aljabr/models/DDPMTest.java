package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;

class DDPMTest {

    @Test
    void samplesConfiguredVectorDimension() {
        try (DDPM model = new DDPM(6, 12, 4);
             Tensor sample = model.sample(3, 1234L)) {
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
        try (DDPM model = new DDPM(4, 8, 5);
             Tensor first = model.sample(2, 99L);
             Tensor second = model.sample(2, 99L)) {
            assertArrayEquals(first.toFloatArray(), second.toFloatArray(), 0f);
        }
    }

    @Test
    void differentSeedsChangeInitialAndReverseNoise() {
        try (DDPM model = new DDPM(4, 8, 5);
             Tensor first = model.sample(2, 11L);
             Tensor second = model.sample(2, 12L)) {
            assertFalse(java.util.Arrays.equals(first.toFloatArray(), second.toFloatArray()));
        }
    }

    @Test
    void rejectsInvalidConstructorDimensionsAndSchedule() {
        assertThrows(IllegalArgumentException.class, () -> new DDPM(0, 8, 4));
        assertThrows(IllegalArgumentException.class, () -> new DDPM(4, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> new DDPM(4, 8, 1));
        assertThrows(IllegalArgumentException.class, () -> new DDPM(4, 8, 0));
    }

    @Test
    void rejectsInvalidSampleBatchSize() {
        try (DDPM model = new DDPM(4, 8, 3)) {
            assertThrows(IllegalArgumentException.class, () -> model.sample(0, 1L));
            assertThrows(IllegalArgumentException.class, () -> model.sample(-1, 1L));
        }
    }

    @Test
    void rejectsTrainingInputWithWrongShape() {
        try (DDPM model = new DDPM(4, 8, 3);
             Tensor wrongWidth = Tensor.of(new float[6], 2, 3)) {
            assertThrows(IllegalArgumentException.class, () -> model.trainingLoss(wrongWidth));
        }
    }
}
