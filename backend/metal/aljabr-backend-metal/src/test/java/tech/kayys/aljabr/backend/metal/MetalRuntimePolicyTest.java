package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

class MetalRuntimePolicyTest {
    @Test
    @ResourceLock("system-properties")
    void requireNativeRejectsForcedCpuMode() {
        String previous = System.getProperty("aljabr.kernel.force.cpu");
        System.setProperty("aljabr.kernel.force.cpu", "true");
        try {
            assertThrows(IllegalStateException.class,
                    () -> new MetalComputeBackend(null, MetalRuntimePolicy.REQUIRE_NATIVE));
        } finally {
            if (previous == null) System.clearProperty("aljabr.kernel.force.cpu");
            else System.setProperty("aljabr.kernel.force.cpu", previous);
        }
    }

    @Test
    @ResourceLock("system-properties")
    void defaultPolicyPreservesForcedCpuCompatibility() {
        String previous = System.getProperty("aljabr.kernel.force.cpu");
        System.setProperty("aljabr.kernel.force.cpu", "true");
        try {
            new MetalComputeBackend().close();
        } finally {
            if (previous == null) System.clearProperty("aljabr.kernel.force.cpu");
            else System.setProperty("aljabr.kernel.force.cpu", previous);
        }
    }
}
