package tech.kayys.aljabr.spi.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.spi.provider.OperationSupport;

class ProviderSessionOperationSupportTest {

    @Test
    void defaultsToUnknownWithoutMakingUnsupportedClaims() {
        ProviderSession session = new ProviderSession() {
            @Override
            public tech.kayys.aljabr.spi.provider.DeviceDescriptor device() {
                return null;
            }

            @Override
            public Map<String, tech.kayys.aljabr.core.backend.ComputeBackend> computeBackends() {
                return Map.of();
            }

            @Override
            public void close() {
            }
        };

        var result = session.operationSupport("tensor.matmul", Map.of("dtype", "f32"));

        assertEquals("tensor.matmul", result.operationId());
        assertEquals(OperationSupport.UNKNOWN, result.support());
        assertEquals(null, result.effectiveBackendId());
    }

    @Test
    void validatesOperationIdAndConstraints() {
        ProviderSession session = new ProviderSession() {
            @Override
            public tech.kayys.aljabr.spi.provider.DeviceDescriptor device() {
                return null;
            }

            @Override
            public Map<String, tech.kayys.aljabr.core.backend.ComputeBackend> computeBackends() {
                return Map.of();
            }

            @Override
            public void close() {
            }
        };

        assertThrows(IllegalArgumentException.class,
                () -> session.operationSupport(" ", Map.of()));
        assertThrows(NullPointerException.class,
                () -> session.operationSupport("tensor.matmul", null));
    }
}
