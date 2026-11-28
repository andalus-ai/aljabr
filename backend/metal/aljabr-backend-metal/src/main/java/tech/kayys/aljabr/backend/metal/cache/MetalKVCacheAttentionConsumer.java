package tech.kayys.aljabr.backend.metal.cache;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheBufferCompatibilityValidator;
import tech.kayys.aljabr.cache.KVCacheBufferLayout;
import tech.kayys.aljabr.cache.KVCacheBufferReadView;
import tech.kayys.aljabr.cache.KVCacheBufferRegion;
import tech.kayys.aljabr.cache.KVCacheMemoryDomain;
import tech.kayys.aljabr.metal.binding.MetalBinding;

/**
 * Direct Metal consumer for contiguous sequence-major borrowed KV buffers.
 *
 * <p>The cache lease remains active throughout the synchronous command-buffer
 * submission and completion wait. K/V MTLBuffer objects are bound directly to
 * the SDPA kernel; no cache staging allocation or K/V copy is performed.</p>
 */
public final class MetalKVCacheAttentionConsumer {
    private static final int MAX_SEQUENCE_LENGTH = 4096;

    private final MetalBinding binding;

    public MetalKVCacheAttentionConsumer(MetalBinding binding) {
        this.binding = Objects.requireNonNull(binding, "binding");
    }

    /**
     * Computes one decode query against the committed prefix visible to lease.
     *
     * @param output caller-owned [queryHeads, headDimension] FLOAT32 output
     * @param query caller-owned [queryHeads, headDimension] FLOAT32 query
     * @param startPosition first visible cache position to attend to
     * @return native bridge status (zero on success)
     */
    public int decode(KVCacheBufferReadView lease, int layer,
            MemorySegment output, MemorySegment query, int queryHeads,
            int startPosition, float scale, float softCap) {
        Objects.requireNonNull(lease, "lease");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(query, "query");
        if (!output.isNative() || !query.isNative()) {
            throw new IllegalArgumentException("Metal attention query and output must use native memory segments");
        }
        // Reject incompatible providers before interpreting their opaque handles.
        KVCacheBufferCompatibilityValidator.validateForConsumer(
                lease, layer, "metal", KVCacheMemoryDomain.DEVICE,
                tech.kayys.aljabr.cache.KVCacheElementType.FLOAT32, binding.deviceName());

        // Kernel-specific layout and offset restrictions are applied below.

        int sequenceLength = lease.visibleSequenceLength();
        if (queryHeads <= 0 || sequenceLength <= 0 || sequenceLength > MAX_SEQUENCE_LENGTH) {
            throw new IllegalArgumentException("Metal flat SDPA requires 1..4096 visible positions");
        }
        if (startPosition < 0 || startPosition >= sequenceLength) {
            throw new IllegalArgumentException("startPosition must be inside the visible prefix");
        }

        KVCacheBufferLayout keyLayout = lease.keyLayout(layer);
        KVCacheBufferLayout valueLayout = lease.valueLayout(layer);
        validateLayout(keyLayout, sequenceLength);
        validateLayout(valueLayout, sequenceLength);
        if (!keyLayout.equals(valueLayout)) {
            throw new IllegalArgumentException("K and V logical layouts must match");
        }
        if (queryHeads % keyLayout.kvHeads() != 0) {
            throw new IllegalArgumentException("Query heads must be divisible by KV heads");
        }
        if (!Float.isFinite(scale) || scale <= 0.0f
                || !Float.isFinite(softCap) || softCap < 0.0f) {
            throw new IllegalArgumentException("Attention scale must be positive and softCap non-negative");
        }
        long queryBytes = Math.multiplyExact(
                Math.multiplyExact((long) queryHeads, keyLayout.headDimension()), Float.BYTES);
        if (query.byteSize() < queryBytes || output.byteSize() < queryBytes) {
            throw new IllegalArgumentException(
                    "Query and output segments must each contain queryHeads * headDimension FLOAT32 values");
        }

        KVCacheBufferRegion keyRegion = lease.keyRegion(layer);
        KVCacheBufferRegion valueRegion = lease.valueRegion(layer);
        validateRegion(keyRegion);
        validateRegion(valueRegion);
        if (!keyRegion.deviceId().equals(valueRegion.deviceId())) {
            throw new IllegalArgumentException("K and V buffers belong to different Metal devices");
        }

        // Establish the provider's write-visibility dependency before dispatch.
        lease.readiness().toCompletableFuture().join();

        return MetalKVCacheBufferAccess.withBuffer(lease, keyRegion, keyView -> {
            if (!keyRegion.deviceId().equals(keyView.deviceId())) {
                throw new IllegalArgumentException("K buffer device identity changed");
            }
            return MetalKVCacheBufferAccess.withBuffer(lease, valueRegion, valueView -> {
                if (!valueView.deviceId().equals(keyView.deviceId())) {
                    throw new IllegalArgumentException("K and V buffer device identities differ");
                }
                return binding.borrowedKvAttention(output, query,
                        keyView.nativeBuffer(), valueView.nativeBuffer(),
                        queryHeads, keyLayout.kvHeads(),
                        keyLayout.headDimension(), Math.toIntExact(keyLayout.sequenceStride()),
                        sequenceLength - 1, startPosition, scale, softCap);
            });
        });
    }

    private static void validateLayout(KVCacheBufferLayout layout, int sequenceLength) {
        if (layout.sequenceLength() != sequenceLength
                || layout.elementStride() != 1
                || layout.headStride() != layout.headDimension()
                || layout.sequenceStride() < (long) layout.kvHeads() * layout.headDimension()) {
            throw new IllegalArgumentException(
                    "Metal flat SDPA requires contiguous [sequence, kvHead, headDimension] layout");
        }
    }

    private static void validateRegion(KVCacheBufferRegion region) {
        if (region.byteOffset() != 0) {
            throw new IllegalArgumentException("Metal flat SDPA requires a zero-based KV region");
        }
    }
}
