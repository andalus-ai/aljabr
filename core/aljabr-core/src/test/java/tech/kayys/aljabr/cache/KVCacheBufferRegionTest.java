package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KVCacheBufferRegionTest {

    private static final class TestHandle implements KVCacheBufferHandle {
    }

    @Test
    void describesHostNativeRegionWithoutExposingAddress() {
        KVCacheBufferRegion region = new KVCacheBufferRegion(
                new TestHandle(),
                KVCacheMemoryDomain.HOST_NATIVE,
                "cpu",
                "",
                KVCacheElementType.FLOAT16,
                16,
                128);

        assertEquals(64, region.elementCount());
        assertEquals("cpu", region.backendId());
        assertEquals(KVCacheMemoryDomain.HOST_NATIVE, region.memoryDomain());
    }

    @Test
    void requiresDeviceIdentityForDeviceRegions() {
        assertThrows(IllegalArgumentException.class, () ->
                new KVCacheBufferRegion(
                        new TestHandle(),
                        KVCacheMemoryDomain.DEVICE,
                        "cuda",
                        "",
                        KVCacheElementType.FLOAT16,
                        0,
                        128));
    }

    @Test
    void rejectsDeviceIdentityForHostRegions() {
        assertThrows(IllegalArgumentException.class, () ->
                new KVCacheBufferRegion(
                        new TestHandle(),
                        KVCacheMemoryDomain.HOST_HEAP,
                        "cpu",
                        "0",
                        KVCacheElementType.FLOAT32,
                        0,
                        128));
    }

    @Test
    void validatesNonNegativeAndElementAlignedByteRanges() {
        assertThrows(IllegalArgumentException.class, () ->
                new KVCacheBufferRegion(
                        new TestHandle(),
                        KVCacheMemoryDomain.HOST_NATIVE,
                        "cpu",
                        "",
                        KVCacheElementType.FLOAT16,
                        1,
                        128));
        assertThrows(IllegalArgumentException.class, () ->
                new KVCacheBufferRegion(
                        new TestHandle(),
                        KVCacheMemoryDomain.HOST_NATIVE,
                        "cpu",
                        "",
                        KVCacheElementType.FLOAT32,
                        0,
                        -4));
    }
}
