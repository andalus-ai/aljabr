package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Tensor;

class MetalComputeBackendOutputOwnershipTest {

    @Test
    void releasesOutputBeforeCpuFallbackWhenNativeDispatchFails() {
        TrackingStorage output = new TrackingStorage();
        AtomicInteger fallbackCalls = new AtomicInteger();

        Tensor result = MetalComputeBackend.dispatchOutput(
                output,
                () -> -1,
                () -> {
                    fallbackCalls.incrementAndGet();
                    return null;
                },
                ignored -> {
                    throw new AssertionError("result factory must not run after native failure");
                });

        assertNull(result);
        assertEquals(1, output.releases.get());
        assertEquals(1, fallbackCalls.get());
    }

    @Test
    void releasesOutputWhenNativeDispatchThrows() {
        TrackingStorage output = new TrackingStorage();

        assertThrows(IllegalStateException.class, () -> MetalComputeBackend.dispatchOutput(
                output,
                () -> {
                    throw new IllegalStateException("native dispatch failed");
                },
                () -> null,
                ignored -> null));

        assertEquals(1, output.releases.get());
    }

    @Test
    void transfersOutputOwnershipToResultOnSuccessfulDispatch() {
        TrackingStorage output = new TrackingStorage();

        Tensor result = MetalComputeBackend.dispatchOutput(
                output,
                () -> 0,
                () -> {
                    throw new AssertionError("CPU fallback must not run after native success");
                },
                ignored -> null);

        assertNull(result);
        assertEquals(0, output.releases.get());
    }

    @Test
    void releasesOutputWhenResultConstructionThrows() {
        TrackingStorage output = new TrackingStorage();

        assertThrows(IllegalArgumentException.class, () -> MetalComputeBackend.dispatchOutput(
                output,
                () -> 0,
                () -> null,
                ignored -> {
                    throw new IllegalArgumentException("cannot construct tensor");
                }));

        assertEquals(1, output.releases.get());
    }

    private static final class TrackingStorage implements HostAccessibleStorage {
        private final AtomicInteger releases = new AtomicInteger();

        @Override
        public MemorySegment segment() {
            return MemorySegment.NULL;
        }

        @Override
        public long sizeBytes() {
            return 0;
        }

        @Override
        public void retain() {
        }

        @Override
        public void release() {
            releases.incrementAndGet();
        }
    }
}
