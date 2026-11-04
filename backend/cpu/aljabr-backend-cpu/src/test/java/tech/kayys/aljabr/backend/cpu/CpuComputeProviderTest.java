package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ServiceLoader;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.spi.compute.ComputeProvider;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderException;
import tech.kayys.aljabr.spi.testkit.ProviderConformance;

class CpuComputeProviderTest {
    @Test
    void serviceLoaderDiscoversCpuProvider() {
        assertEquals(1, ServiceLoader.load(AljabrProvider.class)
                .stream().map(ServiceLoader.Provider::get)
                .filter(CpuComputeProvider.class::isInstance).count());
    }

    @Test
    void providerRequiresInitializationAndOpensCpuSession() throws Exception {
        CpuComputeProvider provider = new CpuComputeProvider();
        assertEquals("andalus/cpu", provider.descriptor().id());
        assertEquals(1, provider.devices().size());
        assertThrows(ProviderException.class, () -> provider.openSession(provider.devices().get(0)));

        provider.initialize(new ProviderContext("test", java.util.Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals("cpu:0", session.device().deviceId());
            assertInstanceOf(CpuBackend.class, session.computeBackends().get(CpuComputeProvider.BACKEND_ID));
        }
        provider.close();
        assertThrows(ProviderException.class, provider::devices);
    }

    @Test
    void initializationIsSingleUseAndCloseIsIdempotent() throws Exception {
        CpuComputeProvider provider = new CpuComputeProvider();
        ProviderContext context = new ProviderContext("test", java.util.Map.of());
        provider.initialize(context);

        assertThrows(ProviderException.class, () -> provider.initialize(context));
        provider.close();
        provider.close();
        assertThrows(ProviderException.class, () -> provider.initialize(context));
    }

    @Test
    void providerCloseClosesOutstandingSessions() throws Exception {
        CpuComputeProvider provider = new CpuComputeProvider();
        provider.initialize(new ProviderContext("test", java.util.Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));

        provider.close();

        assertEquals("cpu:0", session.device().deviceId());
        assertThrows(IllegalStateException.class, session::computeBackends);
        session.close();
        assertThrows(ProviderException.class, () -> provider.openSession(provider.devices().get(0)));
    }

    @Test
    void providerRejectsForeignAndUnknownDevices() throws Exception {
        CpuComputeProvider provider = new CpuComputeProvider();
        provider.initialize(new ProviderContext("test", java.util.Map.of()));
        DeviceDescriptor foreign = new DeviceDescriptor(
                "other/provider", "cpu:0", "Foreign CPU", DeviceKind.CPU,
                null, null, null, null, null, null);
        DeviceDescriptor unknown = new DeviceDescriptor(
                CpuComputeProvider.PROVIDER_ID, "cpu:99", "Unknown CPU", DeviceKind.CPU,
                null, null, null, null, null, null);

        assertThrows(ProviderException.class, () -> provider.openSession(foreign));
        assertThrows(ProviderException.class, () -> provider.openSession(unknown));
        provider.close();
    }

    @Test
    void cpuProviderPassesSharedLifecycleConformance() {
        var result = ProviderConformance.verify(
                CpuComputeProvider::new,
                new ProviderContext("test", java.util.Map.of()));

        assertTrue(result.isConformant(), result.violations().toString());
        assertEquals("andalus/cpu", result.providerId());
    }
}
