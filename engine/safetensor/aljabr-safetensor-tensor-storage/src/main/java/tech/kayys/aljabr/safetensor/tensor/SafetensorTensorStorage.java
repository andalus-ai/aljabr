package tech.kayys.aljabr.safetensor.tensor;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.TensorLayout;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Zero-copy Aljabr host storage backed by a tensor in a Safetensors shard session.
 *
 * <p>The adapter owns one session lease. Retaining this storage creates another
 * adapter reference; the lease is released only after the final storage owner
 * releases it. The session owner may therefore close independently while
 * tensors continue to access the mapped bytes.</p>
 *
 * <p>The returned segment is borrowed. Callers must hold a live storage
 * reference for as long as they use it. Safetensors bytes are little-endian;
 * consumers must use the appropriate byte order when interpreting multi-byte
 * values.</p>
 */
public final class SafetensorTensorStorage implements HostAccessibleStorage {

    private final SafetensorTensor tensor;
    private final SafetensorShardSession.SessionLease lease;
    private final AtomicInteger references = new AtomicInteger(1);

    private SafetensorTensorStorage(
            SafetensorTensor tensor,
            SafetensorShardSession.SessionLease lease) {
        this.tensor = Objects.requireNonNull(tensor, "tensor must not be null");
        this.lease = Objects.requireNonNull(lease, "lease must not be null");
    }

    /**
     * Binds a named tensor in an open session without copying its payload.
     * The session lookup and lease acquisition are failure-safe: if either
     * operation fails, no lease is leaked.
     */
    public static SafetensorTensorStorage open(SafetensorShardSession session, String tensorName) {
        Objects.requireNonNull(session, "session must not be null");
        Objects.requireNonNull(tensorName, "tensorName must not be null");
        SafetensorShardSession.SessionLease lease = session.retain();
        try {
            return new SafetensorTensorStorage(session.tensor(tensorName), lease);
        } catch (RuntimeException | Error failure) {
            lease.close();
            throw failure;
        }
    }

    /** Returns the mapped tensor bytes as a borrowed segment. */
    @Override
    public MemorySegment segment() {
        ensureOpen();
        return tensor.segment();
    }

    @Override
    public long sizeBytes() {
        ensureOpen();
        return tensor.byteLength();
    }

    /** Returns the source tensor's logical shape. */
    public Shape shape() {
        ensureOpen();
        return new Shape(tensor.shape());
    }

    /** Returns the canonical contiguous row-major layout. */
    public TensorLayout layout() {
        return TensorLayout.contiguous(shape());
    }

    /**
     * Maps the source dtype only when Aljabr has an equivalent representation.
     * Unsupported source types fail explicitly rather than silently narrowing
     * or reinterpreting values.
     */
    public DType dtype() {
        ensureOpen();
        return switch (tensor.dtype()) {
            case F32 -> DType.F32;
            case F16 -> DType.F16;
            case BF16 -> DType.BF16;
            case I8 -> DType.I8;
            case I32 -> DType.I32;
            default -> throw new UnsupportedOperationException(
                    "Safetensors dtype " + tensor.dtype()
                            + " has no equivalent Aljabr DType yet");
        };
    }

    /** Number of outstanding ownership references. */
    public int referenceCount() {
        return references.get();
    }

    @Override
    public void retain() {
        while (true) {
            int current = references.get();
            if (current <= 0) {
                throw new IllegalStateException("Safetensors tensor storage has been released");
            }
            if (current == Integer.MAX_VALUE) {
                throw new IllegalStateException("Safetensors tensor storage reference count overflow");
            }
            if (references.compareAndSet(current, current + 1)) {
                return;
            }
        }
    }

    @Override
    public void release() {
        while (true) {
            int current = references.get();
            if (current <= 0) {
                throw new IllegalStateException("Safetensors tensor storage has already been released");
            }
            int next = current - 1;
            if (references.compareAndSet(current, next)) {
                if (next == 0) {
                    lease.close();
                }
                return;
            }
        }
    }

    private void ensureOpen() {
        if (references.get() <= 0) {
            throw new IllegalStateException("Safetensors tensor storage has been released");
        }
    }
}
