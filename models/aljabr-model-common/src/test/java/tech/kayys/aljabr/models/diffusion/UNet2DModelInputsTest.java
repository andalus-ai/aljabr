package tech.kayys.aljabr.models.diffusion;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class UNet2DModelInputsTest {

    @Test
    void closesOwnedInputsExactlyOnceAndAllowsMissingEncoderContext() {
        try (CpuBackend backend = new CpuBackend()) {
            Tensor timestep = TensorFactory.of(new float[] {1.0f}, backend, 1, 1);
            UNet2DModelInputs inputs = new UNet2DModelInputs(timestep, null);

            assertFalse(inputs.isClosed());
            inputs.close();
            inputs.close();

            assertTrue(inputs.isClosed());
            assertThrows(IllegalStateException.class, inputs::timestepEmbedding);
        }
    }

    @Test
    void rejectsAliasedTimestepAndEncoderInputs() {
        try (CpuBackend backend = new CpuBackend();
                Tensor shared = TensorFactory.of(new float[] {1.0f}, backend, 1, 1)) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new UNet2DModelInputs(shared, shared));
        }
    }
}
