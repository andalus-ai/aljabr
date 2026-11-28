package tech.kayys.aljabr.cache.testing;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import tech.kayys.aljabr.cache.KVCacheBufferLayout;
import tech.kayys.aljabr.cache.KVCacheBufferLayoutValidator;
import tech.kayys.aljabr.cache.KVCacheBufferReadView;
import tech.kayys.aljabr.cache.KVCacheBufferRegion;
import tech.kayys.aljabr.cache.KVCacheElementType;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheMemoryDomain;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.cache.KVCacheStorage;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Reusable JUnit contract for backend-aware KV buffer providers.
 *
 * <p>Provider modules supply only allocation and scoped region reading. This
 * contract verifies the shared logical layout, region metadata, actual K/V
 * contents, snapshot semantics, and mutation protection without introducing
 * backend dependencies into the production cache API.</p>
 */
public abstract class KVCacheBufferProviderContractTest {

    protected abstract Provider openProvider();

    protected boolean isProviderAvailable() {
        return true;
    }

    @Test
    public void exportsConformantRegionsAndPreservesBorrowedPrefix() {
        Assumptions.assumeTrue(isProviderAvailable(), "Provider is not available on this host");
        KVCacheLayout layout = new KVCacheLayout(
                5, new int[] {1, 2}, new int[] {2, 3});

        try (Provider provider = openProvider()) {
            KVCacheStorage storage = provider.storage();
            KVCacheSession session = new KVCacheSession(storage);

            for (int layer = 0; layer < layout.layerCount(); layer++) {
                int stride = layout.layerStride(layer);
                float[] keys = values(layer, 0, 2, stride, false);
                float[] values = values(layer, 0, 2, stride, true);
                session.writeRange(layer, 0, 2, keys, 0, values, 0);
            }
            session.commit(2);

            try (KVCacheBufferReadView lease = session.openBufferReadView()) {
                assertEquals(provider.backendId(), lease.backendId());
                assertEquals(2, lease.visibleSequenceLength());
                assertNotNull(lease.readiness());
                lease.readiness().toCompletableFuture().join();

                for (int layer = 0; layer < layout.layerCount(); layer++) {
                    KVCacheBufferLayout expectedLayout = KVCacheBufferLayout.sequenceMajor(
                            2, layout.kvHeads(layer), layout.headDimension(layer));
                    KVCacheBufferRegion key = lease.keyRegion(layer);
                    KVCacheBufferRegion value = lease.valueRegion(layer);

                    assertEquals(provider.memoryDomain(), key.memoryDomain());
                    assertEquals(provider.memoryDomain(), value.memoryDomain());
                    assertEquals(provider.backendId(), key.backendId());
                    assertEquals(provider.backendId(), value.backendId());
                    assertEquals(KVCacheElementType.FLOAT32, key.elementType());
                    assertEquals(KVCacheElementType.FLOAT32, value.elementType());
                    assertEquals((long) 2 * layout.layerStride(layer), key.elementCount());
                    assertEquals((long) 2 * layout.layerStride(layer), value.elementCount());
                    assertEquals((long) 2 * layout.layerStride(layer) * Float.BYTES, key.byteLength());
                    assertEquals((long) 2 * layout.layerStride(layer) * Float.BYTES, value.byteLength());
                    assertEquals(expectedLayout, lease.keyLayout(layer));
                    assertEquals(expectedLayout, lease.valueLayout(layer));
                    KVCacheBufferLayoutValidator.validateView(lease, layer);

                    assertArrayEquals(values(layer, 0, 2, layout.layerStride(layer), false),
                            provider.readRegion(lease, key));
                    assertArrayEquals(values(layer, 0, 2, layout.layerStride(layer), true),
                            provider.readRegion(lease, value));
                }

                // An active lease is a stable prefix snapshot: appending is legal,
                // but changing a position already visible to it is not.
                assertThrows(IllegalStateException.class, () -> storage.write(
                        0, 0, new float[layout.layerStride(0)], 0,
                        new float[layout.layerStride(0)], 0));

                for (int layer = 0; layer < layout.layerCount(); layer++) {
                    int stride = layout.layerStride(layer);
                    session.write(layer, 2,
                            values(layer, 2, 1, stride, false), 0,
                            values(layer, 2, 1, stride, true), 0);
                }
                session.commit(3);

                assertEquals(2, lease.visibleSequenceLength());
                assertEquals(2L * layout.layerStride(0), lease.keyRegion(0).elementCount());
                assertArrayEquals(values(0, 0, 2, layout.layerStride(0), false),
                        provider.readRegion(lease, lease.keyRegion(0)));
            }
        }
    }

    private static float[] values(int layer, int firstPosition, int tokenCount,
            int stride, boolean value) {
        float[] result = new float[tokenCount * stride];
        for (int token = 0; token < tokenCount; token++) {
            for (int element = 0; element < stride; element++) {
                float magnitude = 100.0f * layer + 10.0f * (firstPosition + token) + element + 0.25f;
                result[token * stride + element] = value ? -magnitude : magnitude;
            }
        }
        return result;
    }

    @FunctionalInterface
    public interface RegionReader {
        float[] read(KVCacheBufferReadView lease, KVCacheBufferRegion region);
    }

    public record Provider(
            KVCacheStorage storage,
            String backendId,
            KVCacheMemoryDomain memoryDomain,
            RegionReader regionReader) implements AutoCloseable {

        public Provider {
            if (storage == null || backendId == null || memoryDomain == null || regionReader == null) {
                throw new NullPointerException("Provider fixture fields must not be null");
            }
        }

        public float[] readRegion(KVCacheBufferReadView lease, KVCacheBufferRegion region) {
            return regionReader.read(lease, region);
        }

        @Override
        public void close() {
            storage.close();
        }
    }
}
