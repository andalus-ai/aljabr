package tech.kayys.aljabr.gguf.core;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Runtime-oriented tensor representation used by the in-memory GGUF model API.
 *
 * <p>This is intentionally distinct from {@code gguf.loader.GGUFTensorInfo},
 * which describes a tensor's serialized file layout, and from the public
 * {@code GgufTensorDescriptor}, which exposes format-neutral structural facts.
 * This type carries a resolved {@link GgmlType} and may carry a live
 * {@link MemorySegment}; it must not be used as the public model-inspection
 * contract.
 *
 * @author bhangun
 */
public record GGUFTensorInfo(
        String name,
        long[] shape,
        GgmlType type,
        long dataOffset,
        MemorySegment data
) {

    public GGUFTensorInfo {
        if (name == null || name.isBlank())
            throw new IllegalArgumentException("name");
        if (shape == null || shape.length == 0 || shape.length > 4)
            throw new IllegalArgumentException("shape must have 1-4 dimensions");
        for (long d : shape)
            if (d <= 0)
                throw new IllegalArgumentException("dimension <= 0");
        if (type == null)
            throw new IllegalArgumentException("type");
        if (dataOffset < 0)
            throw new IllegalArgumentException("dataOffset < 0");
    }

    public GGUFTensorInfo(String name, long[] shape, GgmlType type, long dataOffset) {
        this(name, shape, type, dataOffset, null);
    }

    public int nDims() {
        return shape.length;
    }

    public long nElements() {
        long n = 1;
        for (long d : shape) n *= d;
        return n;
    }

    public long rows() {
        return shape.length > 0 ? shape[0] : 1;
    }

    public long cols() {
        if (shape.length < 2) return 1;
        long c = 1;
        for (int i = 1; i < shape.length; i++) c *= shape[i];
        return c;
    }

    public long dataSize() {
        return type.bytesFor(nElements());
    }

    @Override
    public String toString() {
        return "GGUFTensorInfo{name='%s', shape=%s, type=%s, dataOffset=%d}"
                .formatted(name, Arrays.toString(shape), type.label, dataOffset);
    }
}
