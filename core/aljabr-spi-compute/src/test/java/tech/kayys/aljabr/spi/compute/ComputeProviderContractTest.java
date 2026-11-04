package tech.kayys.aljabr.spi.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.reflect.Proxy;
import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.ProviderException;

class ComputeProviderContractTest {

    @Test
    void sessionExposesDeviceScopedBackendsAndExplicitClose() throws Exception {
        DeviceDescriptor device = new DeviceDescriptor(
                "andalus/cpu", "cpu:0", "CPU", DeviceKind.CPU, null, null,
                "generic", "host", null, null);
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> null);
        FakeSession session = new FakeSession(device, Map.of("cpu-default", backend));

        assertSame(device, session.device());
        assertSame(backend, session.computeBackends().get("cpu-default"));
        assertThrows(UnsupportedOperationException.class,
                () -> session.computeBackends().put("other", backend));

        session.close();
        assertEquals(1, session.closeCount);
    }

    private static final class FakeSession implements ProviderSession {
        private final DeviceDescriptor device;
        private final Map<String, ComputeBackend> backends;
        private int closeCount;

        private FakeSession(DeviceDescriptor device, Map<String, ComputeBackend> backends) {
            this.device = device;
            this.backends = Map.copyOf(backends);
        }

        @Override public DeviceDescriptor device() { return device; }
        @Override public Map<String, ComputeBackend> computeBackends() { return backends; }
        @Override public void close() throws ProviderException { closeCount++; }
    }
}