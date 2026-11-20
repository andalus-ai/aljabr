package tech.kayys.aljabr.spi.model.ir;

import java.net.URI;
import java.util.Objects;

/**
 * Lazy reference to tensor bytes in a source container.
 *
 * <p>The offset is never interpreted without its basis. This avoids assuming
 * that every format stores tensor offsets relative to the file, payload area,
 * or an individual shard in the same way.
 */
public record PayloadReference(
        URI source,
        String container,
        long offset,
        long length,
        OffsetBasis offsetBasis) {

    public PayloadReference {
        Objects.requireNonNull(source, "source must not be null");
        if (container == null || container.isBlank()) {
            throw new IllegalArgumentException("container must not be blank");
        }
        if (offset < 0L) {
            throw new IllegalArgumentException("offset must be non-negative");
        }
        if (length < 0L) {
            throw new IllegalArgumentException("length must be non-negative");
        }
        Objects.requireNonNull(offsetBasis, "offsetBasis must not be null");
        try {
            Math.addExact(offset, length);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("payload range overflows long", e);
        }
    }

    public enum OffsetBasis {
        /** Offset is measured from the first byte of the container. */
        CONTAINER,
        /** Offset is measured from the beginning of the format's tensor-data section. */
        DATA_SECTION
    }
}
