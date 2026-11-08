package tech.kayys.aljabr.runtime.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ComputeRuntimeTest {
    @Test
    void opensAndOwnsExplicitCpuProviderSession() {
        ComputeRuntime runtime = ComputeRuntime.open("andalus/cpu");
        try {
            assertEquals("andalus/cpu", runtime.providerId());
            assertEquals("cpu:0", runtime.device().deviceId());
            assertEquals("tech.kayys.aljabr.backend.cpu.CpuBackend",
                    runtime.backend().getClass().getName());
        } finally {
            runtime.close();
        }

        assertThrows(IllegalStateException.class, runtime::backend);
    }

    @Test
    void explicitUnknownProviderFailsInsteadOfFallingBack() {
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> ComputeRuntime.open("andalus/not-installed"));
        assertTrue(failure.getMessage().contains("andalus/not-installed"));
    }
}
