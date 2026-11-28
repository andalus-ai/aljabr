package tech.kayys.aljabr.cache;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;

class KVCacheBufferCompatibilityValidatorTest {

    private static final KVCacheBufferLayout LOGICAL_LAYOUT =
            KVCacheBufferLayout.sequenceMajor(8, 2, 4);

    @Test
    void acceptsMatchingHostConsumerRequirements() {
        KVCacheBufferReadView view = view(
                region(KVCacheMemoryDomain.HOST_HEAP, "cpu", "", KVCacheElementType.FLOAT32),
                region(KVCacheMemoryDomain.HOST_HEAP, "cpu", "", KVCacheElementType.FLOAT32));

        assertDoesNotThrow(() -> KVCacheBufferCompatibilityValidator.validateForConsumer(
                view, 0, "cpu", KVCacheMemoryDomain.HOST_HEAP, KVCacheElementType.FLOAT32, ""));
    }

    @Test
    void acceptsMatchingDeviceConsumerRequirements() {
        KVCacheBufferReadView view = view(
                region(KVCacheMemoryDomain.DEVICE, "metal", "Apple M-series", KVCacheElementType.FLOAT32),
                region(KVCacheMemoryDomain.DEVICE, "metal", "Apple M-series", KVCacheElementType.FLOAT32));

        assertDoesNotThrow(() -> KVCacheBufferCompatibilityValidator.validateForConsumer(
                view, 0, "metal", KVCacheMemoryDomain.DEVICE,
                KVCacheElementType.FLOAT32, "Apple M-series"));
    }

    @Test
    void rejectsDifferentBackendMemoryDomainOrElementEncoding() {
        KVCacheBufferReadView cpuView = view(
                region(KVCacheMemoryDomain.HOST_HEAP, "cpu", "", KVCacheElementType.FLOAT32),
                region(KVCacheMemoryDomain.HOST_HEAP, "cpu", "", KVCacheElementType.FLOAT32));

        assertThrows(IllegalArgumentException.class, () ->
                KVCacheBufferCompatibilityValidator.validateForConsumer(
                        cpuView, 0, "metal", KVCacheMemoryDomain.HOST_HEAP,
                        KVCacheElementType.FLOAT32, ""));
        assertThrows(IllegalArgumentException.class, () ->
                KVCacheBufferCompatibilityValidator.validateForConsumer(
                        cpuView, 0, "cpu", KVCacheMemoryDomain.HOST_NATIVE,
                        KVCacheElementType.FLOAT32, ""));
        assertThrows(IllegalArgumentException.class, () ->
                KVCacheBufferCompatibilityValidator.validateForConsumer(
                        cpuView, 0, "cpu", KVCacheMemoryDomain.HOST_HEAP,
                        KVCacheElementType.FLOAT16, ""));
    }

    @Test
    void rejectsDifferentDeviceIdentityBetweenProviderAndConsumer() {
        KVCacheBufferReadView view = view(
                region(KVCacheMemoryDomain.DEVICE, "metal", "Apple GPU 0", KVCacheElementType.FLOAT32),
                region(KVCacheMemoryDomain.DEVICE, "metal", "Apple GPU 0", KVCacheElementType.FLOAT32));

        assertThrows(IllegalArgumentException.class, () ->
                KVCacheBufferCompatibilityValidator.validateForConsumer(
                        view, 0, "metal", KVCacheMemoryDomain.DEVICE,
                        KVCacheElementType.FLOAT32, "Apple GPU 1"));
    }

    @Test
    void rejectsKeyAndValueRegionsFromDifferentDevices() {
        KVCacheBufferReadView view = view(
                region(KVCacheMemoryDomain.DEVICE, "metal", "Apple GPU 0", KVCacheElementType.FLOAT32),
                region(KVCacheMemoryDomain.DEVICE, "metal", "Apple GPU 1", KVCacheElementType.FLOAT32));

        assertThrows(IllegalArgumentException.class, () ->
                KVCacheBufferCompatibilityValidator.validateForConsumer(
                        view, 0, "metal", KVCacheMemoryDomain.DEVICE,
                        KVCacheElementType.FLOAT32, "Apple GPU 0"));
    }

    private static KVCacheBufferRegion region(
            KVCacheMemoryDomain domain, String backend, String device, KVCacheElementType type) {
        return new KVCacheBufferRegion(new TestHandle(), domain, backend, device, type,
                0, 8L * 2 * 4 * type.bytes());
    }

    private static KVCacheBufferReadView view(
            KVCacheBufferRegion key, KVCacheBufferRegion value) {
        return new KVCacheBufferReadView() {
            @Override public KVCacheLayout layout() { return KVCacheLayout.uniform(1, 16, 2, 4); }
            @Override public int visibleSequenceLength() { return 8; }
            @Override public KVCacheBufferRegion keyRegion(int layer) { checkLayer(layer); return key; }
            @Override public KVCacheBufferRegion valueRegion(int layer) { checkLayer(layer); return value; }
            @Override public KVCacheBufferLayout keyLayout(int layer) { checkLayer(layer); return LOGICAL_LAYOUT; }
            @Override public KVCacheBufferLayout valueLayout(int layer) { checkLayer(layer); return LOGICAL_LAYOUT; }
            @Override public java.util.concurrent.CompletionStage<Void> readiness() {
                return CompletableFuture.completedFuture(null);
            }
            @Override public String backendId() { return key.backendId(); }
            @Override public void close() { }
        };
    }

    private static void checkLayer(int layer) {
        if (layer != 0) throw new IndexOutOfBoundsException("layer");
    }

    private static final class TestHandle implements KVCacheBufferHandle {
    }
}
