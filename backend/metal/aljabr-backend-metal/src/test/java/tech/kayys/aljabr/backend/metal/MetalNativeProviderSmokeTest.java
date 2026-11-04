package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import tech.kayys.aljabr.metal.binding.MetalBinding;
import tech.kayys.aljabr.spi.compute.ComputeProvider;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.ProviderContext;

/**
 * Opt-in native Metal smoke test. Run only on a Mac with the native Metal
 * bridge available using -Daljabr.metal.hardware.tests=true.
 */
class MetalNativeProviderSmokeTest {
    @Test
    @EnabledIfSystemProperty(named = "aljabr.metal.hardware.tests", matches = "true")
    void opensRealNativeMetalProviderSession() throws Exception {
        MetalComputeProvider provider = new MetalComputeProvider();
        try (provider) {
            provider.initialize(new ProviderContext("metal-native-smoke", Map.of()));
            var devices = provider.devices();
            assertFalse(devices.isEmpty(), "A macOS Metal device should be discoverable");

            try (ProviderSession session = provider.openSession(devices.get(0))) {
                assertNotNull(session.device());
                assertTrue(MetalBinding.getInstance().isRuntimeActive(),
                        "A Metal provider session must activate the native Metal runtime");
                assertTrue(session.computeBackends().containsKey(MetalComputeProvider.BACKEND_ID),
                        "The native session must expose the Metal compute backend");
            }
        }
    }
}