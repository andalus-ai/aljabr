package tech.kayys.aljabr.runtime.compute;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.TensorFactory;

class ComputeRuntimeTest {
    @Test
    void opensAndOwnsExplicitCpuProviderSession() {
        ComputeRuntime runtime = ComputeRuntime.open("andalus/cpu");
        try {
            assertEquals("andalus/cpu", runtime.providerId());
            assertEquals("cpu:0", runtime.device().deviceId());
            assertEquals("tech.kayys.aljabr.backend.cpu.CpuBackend",
                    runtime.backend().getClass().getName());
            var tensor = TensorFactory.zeros(runtime.backend(), 2);
            assertEquals(runtime.backend(), tensor.backend());
        } finally {
            runtime.close();
        }

        assertThrows(IllegalStateException.class, runtime::backend);
        assertDoesNotThrow(runtime::close);
    }

    @Test
    void explicitUnknownProviderFailsInsteadOfFallingBack() {
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> ComputeRuntime.open("andalus/not-installed"));
        assertTrue(failure.getMessage().contains("andalus/not-installed"));
    }
}
