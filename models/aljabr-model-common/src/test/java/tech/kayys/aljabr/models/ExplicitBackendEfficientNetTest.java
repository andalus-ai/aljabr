package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class ExplicitBackendEfficientNetTest {

    @Test
    void efficientNetPropagatesBackendToModulesAndState() {
        CpuBackend backend = new CpuBackend();
        NNModule model = new EfficientNet.EfficientNetModel(
                backend, new int[]{8, 16}, new int[]{1, 1}, 4, 32);
        try {
            assertEquals(backend, model.computeBackend());
            model.children().values().forEach(child -> assertEquals(backend, child.computeBackend()));
            model.stateDict().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
        } finally {
            model.close();
        }
    }

    @Test
    void efficientNetRejectsInputFromDifferentBackend() {
        CpuBackend owner = new CpuBackend();
        CpuBackend other = new CpuBackend();
        NNModule model = new EfficientNet.EfficientNetModel(
                owner, new int[]{8}, new int[]{1}, 4, 16);
        Tensor input = TensorFactory.of(new float[3 * 8 * 8], other, 1, 3, 8, 8);
        try {
            assertThrows(IllegalArgumentException.class, () -> model.forward(input));
        } finally {
            input.release();
            model.close();
        }
    }
}
