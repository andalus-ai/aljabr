package tech.kayys.aljabr.runtime.compute;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Proxy;
import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;

import tech.kayys.aljabr.core.tensor.TensorFactory;

class ComputeRuntimeTest {
    @Test
    void opensAndOwnsExplicitCpuProviderSession() {
        ComputeRuntime runtime = ComputeRuntime.open("andalus/cpu");
        try {
            assertEquals("andalus/cpu", runtime.providerId());
            assertEquals("cpu-default", runtime.backendId());
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
    void selectsTheOnlyBackendWhenNoBackendIdIsSpecified() {
        ComputeBackend backend = backendStub();
        assertEquals(backend, ComputeRuntime.selectBackend(Map.of("provider-default", backend), null).backend());
    }

    @Test
    void requiresAnExplicitIdWhenSessionExposesMultipleBackends() {
        ComputeBackend first = backendStub();
        ComputeBackend second = backendStub();
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> ComputeRuntime.selectBackend(
                        Map.of("fast-kernel", first, "portable-kernel", second), null));
        assertTrue(failure.getMessage().contains("select one"));
        assertEquals(first, ComputeRuntime.selectBackend(
                Map.of("fast-kernel", first, "portable-kernel", second), "fast-kernel").backend());
    }

    @Test
    void rejectsUnknownBackendIdsWithAvailableChoices() {
        ComputeBackend backend = backendStub();
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> ComputeRuntime.selectBackend(Map.of("cpu-default", backend), "metal-default"));
        assertTrue(failure.getMessage().contains("cpu-default"));
    }

    private static ComputeBackend backendStub() {
        return (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] {ComputeBackend.class},
                (proxy, method, args) -> null);
    }

    @Test
    void explicitUnknownProviderFailsInsteadOfFallingBack() {
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> ComputeRuntime.open("andalus/not-installed"));
        assertTrue(failure.getMessage().contains("andalus/not-installed"));
    }
}
