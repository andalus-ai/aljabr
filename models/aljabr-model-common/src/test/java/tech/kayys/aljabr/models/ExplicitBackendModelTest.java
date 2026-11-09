package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.nn.NNModule;

class ExplicitBackendModelTest {

    @Test
    void llamaPropagatesBackendToEveryNestedModuleAndStateTensor() {
        CpuBackend backend = new CpuBackend();
        LLaMA.LLaMAModel model = new LLaMA.LLaMAModel(backend, 8, 2, 2, 16, 32, 64, 0.0f);
        try {
            assertEquals(backend, model.computeBackend());
            assertEquals(backend, model.children().get("embed").computeBackend());
            assertEquals(backend, model.children().get("norm").computeBackend());
            assertEquals(backend, model.children().get("lm_head").computeBackend());
            model.children().get("block_0").stateDict().values()
                    .forEach(tensor -> assertEquals(backend, tensor.backend()));
            model.stateDict().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
        } finally {
            model.close();
        }
    }

    @Test
    void llamaRejectsInputOwnedByAnotherBackend() {
        CpuBackend owner = new CpuBackend();
        CpuBackend other = new CpuBackend();
        NNModule model = new LLaMA.LLaMAModel(owner, 8, 2, 1, 16, 32, 64, 0.0f);
        Tensor input = TensorFactory.of(new float[8], other, 1, 1, 8);
        try {
            assertThrows(IllegalArgumentException.class, () -> model.forward(input));
        } finally {
            input.release();
            model.close();
        }
    }
}
