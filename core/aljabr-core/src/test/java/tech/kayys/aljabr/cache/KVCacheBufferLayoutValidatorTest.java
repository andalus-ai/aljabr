package tech.kayys.aljabr.cache;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;

class KVCacheBufferLayoutValidatorTest {

    @Test
    void acceptsARegionWhoseEncodedSpanContainsTheLogicalLayout() {
        KVCacheBufferRegion region = region(KVCacheElementType.FLOAT32, 256);
        KVCacheBufferLayout layout = KVCacheBufferLayout.sequenceMajor(8, 2, 4);

        assertDoesNotThrow(() -> KVCacheBufferLayoutValidator.validateRegion(region, layout));
    }

    @Test
    void acceptsPaddedStridedLayoutWhenItsRequiredSpanFits() {
        KVCacheBufferRegion region = region(KVCacheElementType.FLOAT16, 52);
        KVCacheBufferLayout layout = new KVCacheBufferLayout(2, 2, 2, 8, 3, 1);

        assertDoesNotThrow(() -> KVCacheBufferLayoutValidator.validateRegion(region, layout));
    }

    @Test
    void rejectsRegionThatCannotContainTheDeclaredLayout() {
        KVCacheBufferRegion region = region(KVCacheElementType.FLOAT32, 252);
        KVCacheBufferLayout layout = KVCacheBufferLayout.sequenceMajor(8, 2, 4);

        assertThrows(IllegalArgumentException.class,
                () -> KVCacheBufferLayoutValidator.validateRegion(region, layout));
    }

    @Test
    void rejectsRegionSpanOverflow() {
        KVCacheBufferRegion region = new KVCacheBufferRegion(
                new TestHandle(), KVCacheMemoryDomain.HOST_HEAP, "cpu", "",
                KVCacheElementType.FLOAT32, Long.MAX_VALUE - 3, 4);
        KVCacheBufferLayout layout = KVCacheBufferLayout.sequenceMajor(1, 1, 1);

        assertThrows(IllegalArgumentException.class,
                () -> KVCacheBufferLayoutValidator.validateRegion(region, layout));
    }

    @Test
    void validatesViewBackendAndLogicalDimensions() {
        KVCacheBufferReadView valid = view("cpu",
                region(KVCacheElementType.FLOAT32, 256),
                KVCacheBufferLayout.sequenceMajor(8, 2, 4));

        assertDoesNotThrow(() -> KVCacheBufferLayoutValidator.validateView(valid, 0));

        KVCacheBufferReadView wrongBackend = view("cpu",
                new KVCacheBufferRegion(new TestHandle(), KVCacheMemoryDomain.HOST_HEAP,
                        "other", "", KVCacheElementType.FLOAT32, 0, 256),
                KVCacheBufferLayout.sequenceMajor(8, 2, 4));
        assertThrows(IllegalArgumentException.class,
                () -> KVCacheBufferLayoutValidator.validateView(wrongBackend, 0));

        KVCacheBufferReadView wrongDimensions = view("cpu",
                region(KVCacheElementType.FLOAT32, 256),
                KVCacheBufferLayout.sequenceMajor(8, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> KVCacheBufferLayoutValidator.validateView(wrongDimensions, 0));
    }

    @Test
    void rejectsInvalidLayerAndUndersizedKOrVRegion() {
        KVCacheBufferReadView valid = view("cpu",
                region(KVCacheElementType.FLOAT32, 256),
                KVCacheBufferLayout.sequenceMajor(8, 2, 4));
        assertThrows(IndexOutOfBoundsException.class,
                () -> KVCacheBufferLayoutValidator.validateView(valid, 1));

        KVCacheBufferReadView undersizedValue = new TestReadView(
                region(KVCacheElementType.FLOAT32, 256),
                region(KVCacheElementType.FLOAT32, 252),
                KVCacheBufferLayout.sequenceMajor(8, 2, 4),
                KVCacheBufferLayout.sequenceMajor(8, 2, 4));
        assertThrows(IllegalArgumentException.class,
                () -> KVCacheBufferLayoutValidator.validateView(undersizedValue, 0));
    }

    private static KVCacheBufferRegion region(KVCacheElementType type, long bytes) {
        return new KVCacheBufferRegion(new TestHandle(), KVCacheMemoryDomain.HOST_HEAP,
                "cpu", "", type, 0, bytes);
    }

    private static KVCacheBufferReadView view(
            String backend, KVCacheBufferRegion region, KVCacheBufferLayout layout) {
        return new TestReadView(region, region, layout, layout, backend);
    }

    private static final class TestHandle implements KVCacheBufferHandle {
    }

    private static final class TestReadView implements KVCacheBufferReadView {
        private final KVCacheBufferRegion key;
        private final KVCacheBufferRegion value;
        private final KVCacheBufferLayout keyLayout;
        private final KVCacheBufferLayout valueLayout;
        private final String backend;

        private TestReadView(KVCacheBufferRegion key, KVCacheBufferRegion value,
                KVCacheBufferLayout keyLayout, KVCacheBufferLayout valueLayout) {
            this(key, value, keyLayout, valueLayout, "cpu");
        }

        private TestReadView(KVCacheBufferRegion key, KVCacheBufferRegion value,
                KVCacheBufferLayout keyLayout, KVCacheBufferLayout valueLayout, String backend) {
            this.key = key;
            this.value = value;
            this.keyLayout = keyLayout;
            this.valueLayout = valueLayout;
            this.backend = backend;
        }

        @Override public KVCacheLayout layout() { return KVCacheLayout.uniform(1, 16, 2, 4); }
        @Override public int visibleSequenceLength() { return 8; }
        @Override public KVCacheBufferRegion keyRegion(int layer) { checkLayer(layer); return key; }
        @Override public KVCacheBufferRegion valueRegion(int layer) { checkLayer(layer); return value; }
        @Override public KVCacheBufferLayout keyLayout(int layer) { checkLayer(layer); return keyLayout; }
        @Override public KVCacheBufferLayout valueLayout(int layer) { checkLayer(layer); return valueLayout; }
        @Override public java.util.concurrent.CompletionStage<Void> readiness() {
            return CompletableFuture.completedFuture(null);
        }
        @Override public String backendId() { return backend; }
        @Override public void close() { }
        private static void checkLayer(int layer) {
            if (layer != 0) throw new IndexOutOfBoundsException("layer");
        }
    }
}
