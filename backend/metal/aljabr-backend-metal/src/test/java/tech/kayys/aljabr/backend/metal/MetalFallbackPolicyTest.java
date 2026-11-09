package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

class MetalFallbackPolicyTest {
    @Test
    @ResourceLock("system-properties")
    void strictPolicyRejectsOperationsWhenCpuModeIsForced() {
        String previous = System.getProperty("aljabr.kernel.force.cpu");
        System.setProperty("aljabr.kernel.force.cpu", "true");
        try {
            MetalComputeBackend backend = new MetalComputeBackend(
                    null,
                    MetalRuntimePolicy.ALLOW_CPU_FALLBACK,
                    MetalFallbackPolicy.FAIL_IF_UNSUPPORTED);
            UnsupportedOperationException failure = assertThrows(
                    UnsupportedOperationException.class,
                    () -> backend.add(null, null));
            assertTrue(failure.getMessage().contains("'add'"));
        } finally {
            if (previous == null) System.clearProperty("aljabr.kernel.force.cpu");
            else System.setProperty("aljabr.kernel.force.cpu", previous);
        }
    }
}
