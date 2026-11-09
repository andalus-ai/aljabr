package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Proxy;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderException;

class MetalComputeProviderTest {
    @Test
    void serviceLoaderDiscoversProviderWithoutCreatingBackend() {
        AtomicInteger backendCreations = new AtomicInteger();
        MetalComputeProvider provider = new MetalComputeProvider(() -> true, () -> {
            backendCreations.incrementAndGet();
            return inertBackend();
        });

        assertEquals("andalus/metal", provider.descriptor().id());
        assertEquals(0, backendCreations.get());
        assertEquals(1, ServiceLoader.load(AljabrProvider.class).stream()
                .map(ServiceLoader.Provider::get)
                .filter(MetalComputeProvider.class::isInstance).count());
        assertEquals(0, backendCreations.get());
    }

    @Test
    void providerOpensAndClosesDeviceScopedSession() throws Exception {
        ComputeBackend backend = inertBackend();
        MetalComputeProvider provider = new MetalComputeProvider(() -> true, () -> backend);
        provider.initialize(new ProviderContext("test", Map.of()));

        DeviceDescriptor device = provider.devices().get(0);
        assertEquals(DeviceKind.GPU, device.kind());
        assertEquals("Apple", device.vendor());
        try (ProviderSession session = provider.openSession(device)) {
            assertEquals(device, session.device());
            assertEquals(backend, session.computeBackends().get(MetalComputeProvider.BACKEND_ID));
        }
    }

    @Test
    void sessionCloseReleasesItsOwnedBackendExactlyOnce() throws Exception {
        AtomicInteger closeCalls = new AtomicInteger();
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("close")) {
                        closeCalls.incrementAndGet();
                        return null;
                    }
                    if (method.getDeclaringClass() == Object.class) {
                        return switch (method.getName()) {
                            case "toString" -> "tracked-metal-backend";
                            case "hashCode" -> System.identityHashCode(proxy);
                            case "equals" -> proxy == args[0];
                            default -> null;
                        };
                    }
                    throw new UnsupportedOperationException("No hardware execution in provider contract test");
                });
        MetalComputeProvider provider = new MetalComputeProvider(() -> true, () -> backend);
        provider.initialize(new ProviderContext("test", Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));

        session.close();
        session.close();

        assertEquals(1, closeCalls.get());
        provider.close();
    }

    @Test
    void providerRejectsUnsupportedPlatformAndForeignDevice() throws Exception {
        MetalComputeProvider unsupported = new MetalComputeProvider(() -> false, MetalComputeProviderTest::inertBackend);
        unsupported.initialize(new ProviderContext("test", Map.of()));
        assertTrue(unsupported.devices().isEmpty());
        assertThrows(ProviderException.class, () -> unsupported.openSession(
                new DeviceDescriptor("andalus/metal", "metal:0", "Apple Metal GPU",
                        DeviceKind.GPU, null, null, "Apple", "test", null, null)));

        MetalComputeProvider provider = new MetalComputeProvider(() -> true, MetalComputeProviderTest::inertBackend);
        provider.initialize(new ProviderContext("test", Map.of()));
        DeviceDescriptor foreign = new DeviceDescriptor("other/provider", "metal:0", "Foreign GPU",
                DeviceKind.GPU, null, null, null, null, null, null);
        assertThrows(ProviderException.class, () -> provider.openSession(foreign));
    }

    @Test
    void providerRequiresInitializationAndRejectsUseAfterClose() throws Exception {
        MetalComputeProvider provider = new MetalComputeProvider(() -> true, MetalComputeProviderTest::inertBackend);
        DeviceDescriptor device = provider.devices().get(0);
        assertThrows(ProviderException.class, () -> provider.openSession(device));
        provider.initialize(new ProviderContext("test", Map.of()));
        provider.close();
        assertThrows(ProviderException.class, provider::devices);
        assertThrows(ProviderException.class, () -> provider.openSession(device));
    }


    @Test
    void providerCloseInvalidatesOutstandingSessionsAndIsIdempotent() throws Exception {
        ComputeBackend backend = inertBackend();
        MetalComputeProvider provider = new MetalComputeProvider(() -> true, () -> backend);
        provider.initialize(new ProviderContext("test", Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));

        provider.close();
        provider.close();

        assertEquals("metal:0", session.device().deviceId());
        assertThrows(IllegalStateException.class, session::computeBackends);
        session.close();
        assertThrows(ProviderException.class, () -> provider.openSession(provider.devices().get(0)));
    }

    @Test
    void providerInitializationIsSingleUse() throws Exception {
        MetalComputeProvider provider = new MetalComputeProvider(() -> true, MetalComputeProviderTest::inertBackend);
        ProviderContext context = new ProviderContext("test", Map.of());
        provider.initialize(context);
        assertThrows(ProviderException.class, () -> provider.initialize(context));
        provider.close();
    }

    private static ComputeBackend inertBackend() {
        return (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class }, (proxy, method, args) -> {
                    if (method.getDeclaringClass() == Object.class) {
                        return switch (method.getName()) {
                            case "toString" -> "inert-metal-backend";
                            case "hashCode" -> System.identityHashCode(proxy);
                            case "equals" -> proxy == args[0];
                            default -> null;
                        };
                    }
                    throw new UnsupportedOperationException("No hardware execution in provider contract test");
                });
    }
}