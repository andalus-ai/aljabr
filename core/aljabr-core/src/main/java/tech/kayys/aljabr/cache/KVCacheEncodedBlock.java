package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Immutable encoded representation of one KV-cache block.
 *
 * <p>The payload contains algorithm-specific encoded values. Metadata contains
 * every per-block parameter required to decode those values, such as scales,
 * codebook references, transform parameters, or residual information. Both
 * arrays are defensively copied on construction and access.</p>
 *
 * <p>Codec identity and version are explicit so persisted or transferred
 * blocks cannot silently be decoded by an incompatible implementation.</p>
 */
public final class KVCacheEncodedBlock {
    private final String codecId;
    private final int codecVersion;
    private final int elementCount;
    private final byte[] payload;
    private final byte[] metadata;

    public KVCacheEncodedBlock(
            String codecId,
            int codecVersion,
            int elementCount,
            byte[] payload,
            byte[] metadata) {
        this.codecId = requireText(codecId, "codecId");
        if (codecVersion <= 0) {
            throw new IllegalArgumentException("codecVersion must be positive");
        }
        if (elementCount <= 0) {
            throw new IllegalArgumentException("elementCount must be positive");
        }
        Objects.requireNonNull(payload, "payload");
        Objects.requireNonNull(metadata, "metadata");
        if (payload.length == 0) {
            throw new IllegalArgumentException("payload must not be empty");
        }
        this.codecVersion = codecVersion;
        this.elementCount = elementCount;
        this.payload = payload.clone();
        this.metadata = metadata.clone();
    }

    public String codecId() {
        return codecId;
    }

    public int codecVersion() {
        return codecVersion;
    }

    /** Number of original float elements represented by this block. */
    public int elementCount() {
        return elementCount;
    }

    /** Returns a defensive copy of the encoded payload. */
    public byte[] payload() {
        return payload.clone();
    }

    /** Returns a defensive copy of the decoder metadata. */
    public byte[] metadata() {
        return metadata.clone();
    }

    public int payloadSizeBytes() {
        return payload.length;
    }

    public int metadataSizeBytes() {
        return metadata.length;
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
