package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.nn.NNModule;

class ExplicitBackendResNetTest {

    @Test
    void resnetFactoriesPropagateBackendToEveryModuleAndStateTensor() {
        CpuBackend backend = new CpuBackend();
        NNModule model = ResNet.resnet18(backend, 10);
        try {
            assertEquals(backend, model.computeBackend());
            model.children().values().forEach(child -> assertEquals(backend, child.computeBackend()));
            model.stateDict().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
        } finally {
            model.close();
        }
    }

    @Test
    void resnetRejectsInputFromAnotherBackendBeforeExecutingLayers() {
        CpuBackend owner = new CpuBackend();
        CpuBackend other = new CpuBackend();
        NNModule model = ResNet.resnet18(owner, 10);
        Tensor input = TensorFactory.of(new float[3 * 8 * 8], other, 1, 3, 8, 8);
        try {
            assertThrows(IllegalArgumentException.class, () -> model.forward(input));
        } finally {
            input.release();
            model.close();
        }
    }
}
