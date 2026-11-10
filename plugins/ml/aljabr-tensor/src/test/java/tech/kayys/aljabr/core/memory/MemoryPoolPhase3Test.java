package tech.kayys.aljabr.core.memory;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Phase 3: Session-scoped Arena Memory Pool.
 */
class MemoryPoolPhase3Test {

    // ── OffHeapBufferPool ──────────────────────────────────────────────────────

    @Test
    void poolAcquireAndReleaseReusesSameMemory() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment seg1 = pool.acquire(256);
            assertNotNull(seg1);
            assertTrue(seg1.byteSize() >= 256);

            pool.release(seg1);
            assertEquals(1, pool.misses()); // first acquire is always a miss

            MemorySegment seg2 = pool.acquire(256);
            assertEquals(1, pool.hits()); // second acquire must be a hit
            assertEquals(seg1.address(), seg2.address(), "same native address should be reused");
        }
    }

    @Test
    void poolBucketAlignsSizeToPowerOfTwo() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            // Request 100 bytes — should be rounded up to the 128-byte bucket.
            MemorySegment seg = pool.acquire(100);
            assertTrue(seg.byteSize() >= 128, "size should be rounded up to next power of 2");
        }
    }

    @Test
    void poolLargeAllocationBypassesBuckets() {
        // 128 MiB exceeds MAX_BUCKET_BYTES — must not be pooled.
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            long bigSize = 128L * 1024 * 1024;
            MemorySegment seg = pool.acquire(bigSize);
            assertTrue(seg.byteSize() >= bigSize);
            pool.release(seg); // should be silently ignored (no bucket for this size)
            // A second acquire should still be a miss (not recycled).
            pool.acquire(bigSize);
            assertEquals(2, pool.misses());
            assertEquals(0, pool.hits());
        }
    }

    @Test
    void poolStatsReflectHitsAndMisses() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment seg = pool.acquire(512);
            pool.release(seg);
            pool.acquire(512);

            String stats = pool.stats();
            assertTrue(stats.contains("hits=1"), stats);
            assertTrue(stats.contains("misses=1"), stats);
        }
    }

    // ── ManagedArena ──────────────────────────────────────────────────────────

    @Test
    void managedArenaClosesOnlyWhenRefCountReachesZero() {
        ManagedArena ma = ManagedArena.ofShared();
        assertEquals(1, ma.refCount());

        ma.retain();
        assertEquals(2, ma.refCount());
        assertTrue(ma.isOpen());

        ma.close(); // drops consumer reference
        assertEquals(1, ma.refCount());
        assertTrue(ma.isOpen(), "should still be open — producer holds a reference");

        ma.close(); // drops producer reference → frees memory
        assertEquals(0, ma.refCount());
        assertFalse(ma.isOpen());
    }

    @Test
    void managedArenaAllocatesMemory() {
        try (ManagedArena ma = ManagedArena.ofShared()) {
            MemorySegment seg = ma.allocate(1024);
            assertEquals(1024, seg.byteSize());
        }
    }

    @Test
    void managedArenaRetainOnClosedThrows() {
        ManagedArena ma = ManagedArena.ofShared();
        ma.close();
        assertThrows(IllegalStateException.class, ma::retain);
    }

    // ── InferenceSession ──────────────────────────────────────────────────────

    @Test
    void inferenceSessionClosesPoolOnExit() {
        OffHeapBufferPool[] poolRef = { null };
        try (InferenceSession session = InferenceSession.of(pool -> {
            poolRef[0] = pool;
            return new DummyBackend(pool);
        })) {
            assertNotNull(session.backend());
            assertNotNull(session.pool());
        }
        // Pool arena is closed — further allocations would fail.
        // We only check that close() didn't throw.
    }

    @Test
    void inferenceSessionRejectsCallsAfterClose() {
        InferenceSession session = InferenceSession.of(
                pool -> new DummyBackend(pool));
        session.close();
        assertThrows(IllegalStateException.class, session::backend);
    }

    // A simple stub to avoid circular dependency on CpuBackend
    static class DummyBackend implements tech.kayys.aljabr.core.backend.ComputeBackend {
        public DummyBackend(OffHeapBufferPool pool) {
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor add(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.Tensor b) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor sub(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.Tensor b) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor mul(tech.kayys.aljabr.core.tensor.Tensor a,
                float scalar) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor mul(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.Tensor b) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor div(tech.kayys.aljabr.core.tensor.Tensor a,
                float scalar) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor div(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.Tensor b) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor addScalar(tech.kayys.aljabr.core.tensor.Tensor a,
                float scalar) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor matmul(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.Tensor b) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor reshape(tech.kayys.aljabr.core.tensor.Tensor a,
                long... shape) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor attention(tech.kayys.aljabr.core.tensor.Tensor Q,
                tech.kayys.aljabr.core.tensor.Tensor K, tech.kayys.aljabr.core.tensor.Tensor V) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor softmax(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor slice(tech.kayys.aljabr.core.tensor.Tensor a,
                long[] offsets, long[] sizes) {
            return null;
        }

        @Override
        public java.util.List<tech.kayys.aljabr.core.tensor.Tensor> split(
                tech.kayys.aljabr.core.tensor.Tensor a, int axis, int parts) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor pow(tech.kayys.aljabr.core.tensor.Tensor a,
                float exponent) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor mean(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor abs(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor crossEntropy(tech.kayys.aljabr.core.tensor.Tensor pred,
                tech.kayys.aljabr.core.tensor.Tensor target) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor binaryCrossEntropy(
                tech.kayys.aljabr.core.tensor.Tensor pred, tech.kayys.aljabr.core.tensor.Tensor target) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor cast(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.DType dtype) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor to(tech.kayys.aljabr.core.tensor.Tensor a,
                tech.kayys.aljabr.core.tensor.DeviceType device) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor zerosLike(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor sqrt(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor relu(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor sigmoid(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor tanh(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor log(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor exp(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor silu(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor flatten(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor unsqueeze(tech.kayys.aljabr.core.tensor.Tensor a,
                int dim) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor squeeze(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor transpose(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor transpose(tech.kayys.aljabr.core.tensor.Tensor a, int d0,
                int d1) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor gelu(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor softmax(tech.kayys.aljabr.core.tensor.Tensor a, int dim) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor logSoftmax(tech.kayys.aljabr.core.tensor.Tensor a,
                int dim) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor mean(tech.kayys.aljabr.core.tensor.Tensor a, int dim,
                boolean keepDim) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor sum(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor sum(tech.kayys.aljabr.core.tensor.Tensor a, int dim,
                boolean keepDim) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor max(tech.kayys.aljabr.core.tensor.Tensor a) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor layerNorm(tech.kayys.aljabr.core.tensor.Tensor input,
                long[] normalizedShape, tech.kayys.aljabr.core.tensor.Tensor weight,
                tech.kayys.aljabr.core.tensor.Tensor bias, float eps) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor rmsNorm(tech.kayys.aljabr.core.tensor.Tensor input,
                tech.kayys.aljabr.core.tensor.Tensor weight, float eps) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor batchNorm(tech.kayys.aljabr.core.tensor.Tensor input,
                tech.kayys.aljabr.core.tensor.Tensor weight, tech.kayys.aljabr.core.tensor.Tensor bias,
                tech.kayys.aljabr.core.tensor.Tensor runningMean,
                tech.kayys.aljabr.core.tensor.Tensor runningVar, boolean training, float momentum, float eps) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor conv2d(tech.kayys.aljabr.core.tensor.Tensor input,
                tech.kayys.aljabr.core.tensor.Tensor weight, tech.kayys.aljabr.core.tensor.Tensor bias,
                int stride, int padding, int dilation, int groups) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor maxPool2d(tech.kayys.aljabr.core.tensor.Tensor input,
                int kernelSize, int stride, int padding) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor adaptiveAvgPool2d(
                tech.kayys.aljabr.core.tensor.Tensor input, int outputH, int outputW) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor dropout(tech.kayys.aljabr.core.tensor.Tensor input,
                float p, boolean training) {
            return null;
        }

        @Override
        public tech.kayys.aljabr.core.tensor.Tensor embedding(tech.kayys.aljabr.core.tensor.Tensor weight,
                tech.kayys.aljabr.core.tensor.Tensor input, long paddingIdx) {
            return null;
        }

        @Override
        public long numel(tech.kayys.aljabr.core.tensor.Tensor a) {
            return 0;
        }
    }
}
