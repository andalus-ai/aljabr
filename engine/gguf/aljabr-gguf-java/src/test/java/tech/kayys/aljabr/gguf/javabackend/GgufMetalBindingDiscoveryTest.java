package tech.kayys.aljabr.gguf.javabackend;

import java.util.logging.Logger;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

class GgufMetalBindingDiscoveryTest {
    @Test
    void explicitDisableSkipsOptionalMetalBindingDiscovery() {
        String previous = System.getProperty("aljabr.metal.disable");
        try {
            System.setProperty("aljabr.metal.disable", "true");

            GgufMetalBindingDiscovery.Result result =
                    GgufMetalBindingDiscovery.discover(Logger.getLogger("test"));

            assertFalse(result.enabled());
            assertNull(result.instance());
            assertNull(result.q4kHandle());
            assertNull(result.fa4Handle());
            assertNull(result.decodeStepHandle());
        } finally {
            if (previous == null) {
                System.clearProperty("aljabr.metal.disable");
            } else {
                System.setProperty("aljabr.metal.disable", previous);
            }
        }
    }
}
