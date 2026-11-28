package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Validates the logical geometry exported by a borrowed KV buffer view.
 *
 * <p>This is a metadata conformance check, not a proof that an opaque handle
 * is readable by a particular backend. Consumers must still check memory
 * domain, device identity, handle compatibility, and readiness.</p>
 */
public final class KVCacheBufferLayoutValidator {

    private KVCacheBufferLayoutValidator() {
    }

    /**
     * Validates one region against its declared logical layout.
     *
     * @throws IllegalArgumentException when the encoded region cannot contain
     *         the complete logical tensor described by the layout
     */
    public static void validateRegion(
            KVCacheBufferRegion region,
            KVCacheBufferLayout logicalLayout) {
        Objects.requireNonNull(region, "region");
        Objects.requireNonNull(logicalLayout, "logicalLayout");

        final long requiredBytes;
        try {
            requiredBytes = logicalLayout.minimumByteLength(region.elementType().bytes());
            Math.addExact(region.byteOffset(), region.byteLength());
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("KV buffer region or layout span overflows", overflow);
        }

        if (requiredBytes > region.byteLength()) {
            throw new IllegalArgumentException(
                    "KV buffer region is too small for its declared layout: requires "
                            + requiredBytes + " bytes but exposes " + region.byteLength());
        }
    }

    /**
     * Validates both K and V metadata for a layer against the owning view.
     * This should be called before interpreting either opaque region.
     *
     * @throws IllegalArgumentException when provider metadata is inconsistent
     */
    public static void validateView(KVCacheBufferReadView view, int layer) {
        Objects.requireNonNull(view, "view");
        KVCacheLayout cacheLayout = Objects.requireNonNull(view.layout(), "view.layout");
        if (layer < 0 || layer >= cacheLayout.layerCount()) {
            throw new IndexOutOfBoundsException("KV layer index out of range: " + layer);
        }

        int visibleLength = view.visibleSequenceLength();
        if (visibleLength < 0 || visibleLength > cacheLayout.maxSequenceLength()) {
            throw new IllegalArgumentException("Visible KV prefix exceeds the declared cache capacity");
        }
        int expectedHeads = cacheLayout.kvHeads(layer);
        int expectedHeadDimension = cacheLayout.headDimension(layer);
        String expectedBackend = view.backendId();
        if (expectedBackend == null || expectedBackend.isBlank()) {
            throw new IllegalArgumentException("Read view backend identifier must not be blank");
        }

        validateViewRegion("key", expectedBackend, visibleLength,
                expectedHeads, expectedHeadDimension,
                view.keyRegion(layer), view.keyLayout(layer));
        validateViewRegion("value", expectedBackend, visibleLength,
                expectedHeads, expectedHeadDimension,
                view.valueRegion(layer), view.valueLayout(layer));
    }

    private static void validateViewRegion(
            String name,
            String expectedBackend,
            int visibleLength,
            int expectedHeads,
            int expectedHeadDimension,
            KVCacheBufferRegion region,
            KVCacheBufferLayout logicalLayout) {
        Objects.requireNonNull(region, name + "Region");
        Objects.requireNonNull(logicalLayout, name + "Layout");

        if (!expectedBackend.equals(region.backendId())) {
            throw new IllegalArgumentException(name + " region backend does not match its read view");
        }
        if (logicalLayout.sequenceLength() != visibleLength
                || logicalLayout.kvHeads() != expectedHeads
                || logicalLayout.headDimension() != expectedHeadDimension) {
            throw new IllegalArgumentException(name + " layout dimensions do not match the read view");
        }
        validateRegion(region, logicalLayout);
    }
}
