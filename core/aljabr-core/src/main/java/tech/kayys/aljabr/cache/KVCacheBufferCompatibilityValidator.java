package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Validates whether a borrowed KV view can be consumed by a particular backend.
 *
 * <p>A valid region descriptor is not universally accessible. Consumers must
 * declare the backend, memory domain, element encoding, and device identity
 * they support before interpreting provider-owned handles.</p>
 */
public final class KVCacheBufferCompatibilityValidator {

    private KVCacheBufferCompatibilityValidator() {
    }

    /**
     * Validates both K and V regions against one consumer's access requirements.
     *
     * @param view active borrowed view
     * @param layer layer to consume
     * @param expectedBackendId consumer backend identifier
     * @param expectedMemoryDomain required allocation domain
     * @param expectedElementType required encoded element type
     * @param expectedDeviceId exact device identity, or empty for host memory
     * @throws IllegalArgumentException if the provider's regions cannot be
     *         consumed directly by this consumer
     */
    public static void validateForConsumer(
            KVCacheBufferReadView view,
            int layer,
            String expectedBackendId,
            KVCacheMemoryDomain expectedMemoryDomain,
            KVCacheElementType expectedElementType,
            String expectedDeviceId) {
        Objects.requireNonNull(view, "view");
        Objects.requireNonNull(expectedBackendId, "expectedBackendId");
        Objects.requireNonNull(expectedMemoryDomain, "expectedMemoryDomain");
        Objects.requireNonNull(expectedElementType, "expectedElementType");
        Objects.requireNonNull(expectedDeviceId, "expectedDeviceId");
        if (expectedBackendId.isBlank()) {
            throw new IllegalArgumentException("Consumer backend identifier must not be blank");
        }
        if (expectedMemoryDomain == KVCacheMemoryDomain.DEVICE && expectedDeviceId.isBlank()) {
            throw new IllegalArgumentException("Device consumers must declare an exact device identity");
        }
        if (expectedMemoryDomain != KVCacheMemoryDomain.DEVICE && !expectedDeviceId.isBlank()) {
            throw new IllegalArgumentException("Host consumers must use an empty device identity");
        }

        KVCacheBufferLayoutValidator.validateView(view, layer);
        validateRegion("key", view.keyRegion(layer), expectedBackendId,
                expectedMemoryDomain, expectedElementType, expectedDeviceId);
        validateRegion("value", view.valueRegion(layer), expectedBackendId,
                expectedMemoryDomain, expectedElementType, expectedDeviceId);
    }

    private static void validateRegion(
            String name,
            KVCacheBufferRegion region,
            String expectedBackendId,
            KVCacheMemoryDomain expectedMemoryDomain,
            KVCacheElementType expectedElementType,
            String expectedDeviceId) {
        if (!expectedBackendId.equals(region.backendId())) {
            throw new IllegalArgumentException(name + " region belongs to backend "
                    + region.backendId() + ", not consumer backend " + expectedBackendId);
        }
        if (region.memoryDomain() != expectedMemoryDomain) {
            throw new IllegalArgumentException(name + " region memory domain "
                    + region.memoryDomain() + " is incompatible with consumer domain "
                    + expectedMemoryDomain);
        }
        if (region.elementType() != expectedElementType) {
            throw new IllegalArgumentException(name + " region element type "
                    + region.elementType() + " is incompatible with consumer element type "
                    + expectedElementType);
        }
        if (!expectedDeviceId.equals(region.deviceId())) {
            throw new IllegalArgumentException(name + " region device identity does not match consumer");
        }
    }
}
