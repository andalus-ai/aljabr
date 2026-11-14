package tech.kayys.aljabr.backend.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.ServiceLoader;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderException;

class HatComputeProviderTest {
    @Test
    void serviceLoaderDiscoversHatProvider() {
        assertEquals(1, ServiceLoader.load(AljabrProvider.class)
                .stream().map(ServiceLoader.Provider::get)
                .filter(HatComputeProvider.class::isInstance).count());
    }

    @Test
    void providerRequiresInitializationAndOpensHatSession() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        assertEquals("andalus/hat-java", provider.descriptor().id());
        assertEquals(DeviceKind.CPU, provider.devices().get(0).kind());
        assertEquals("hat-java:0", provider.devices().get(0).deviceId());
        assertThrows(ProviderException.class,
                () -> provider.openSession(provider.devices().get(0)));

        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(HatComputeProvider.BACKEND_ID,
                    session.computeBackends().keySet().iterator().next());
            assertInstanceOf(HatComputeBackend.class,
                    session.computeBackends().get(HatComputeProvider.BACKEND_ID));
        }
        provider.close();
        assertThrows(ProviderException.class, provider::devices);
    }

    @Test
    void operationSupportDistinguishesNativeHatMatmulFromCpuFallback() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));
        try {
            assertEquals(OperationSupport.NATIVE,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F32", "rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F16", "rank", "2")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F32")).support());

            var fallback = session.operationSupport("tensor.add", Map.of());
            assertEquals(OperationSupport.DELEGATED, fallback.support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID, fallback.effectiveBackendId());
        } finally {
            provider.close();
        }
        assertThrows(IllegalStateException.class, session::computeBackends);
    }

    @Test
    void providerCloseClosesOutstandingSessions() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));

        provider.close();

        assertEquals("hat-java:0", session.device().deviceId());
        assertThrows(IllegalStateException.class, session::computeBackends);
        session.close();
        assertThrows(ProviderException.class, () -> provider.openSession(provider.devices().get(0)));
    }

    @Test
    void sessionExposesOnlyTheProviderOwnedBackendInstance() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            ComputeBackend backend = session.computeBackends().get(HatComputeProvider.BACKEND_ID);
            assertInstanceOf(HatComputeBackend.class, backend);
            assertEquals(1, session.computeBackends().size());
        } finally {
            provider.close();
        }
    }
}
