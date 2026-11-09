package tech.kayys.aljabr.core.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class ExplicitBackendModuleTest {

    @Test
    void linearParametersAndOutputsUseTheConfiguredBackend() {
        CpuBackend backend = new CpuBackend();
        Linear linear = new Linear(backend, 2, 3);
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 1, 2);
        Tensor output = null;
        try {
            assertEquals(backend, linear.computeBackend());
            assertEquals(backend, linear.parameters().get("weight").backend());
            output = linear.forward(input);
            assertEquals(backend, output.backend());
        } finally {
            if (output != null) {
                output.release();
            }
            input.release();
            linear.close();
        }
    }

    @Test
    void linearRejectsInputFromAnotherBackend() {
        CpuBackend configuredBackend = new CpuBackend();
        CpuBackend inputBackend = new CpuBackend();
        Linear linear = new Linear(configuredBackend, 2, 3);
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, inputBackend, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> linear.forward(input));
        } finally {
            input.release();
            linear.close();
        }
    }
}
