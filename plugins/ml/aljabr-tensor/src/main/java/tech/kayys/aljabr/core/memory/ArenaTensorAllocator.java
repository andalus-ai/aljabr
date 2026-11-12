package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Host-memory {@link TensorAllocator} that gives each allocation its own
 * explicitly closable shared FFM arena.
 *
 * <p>This is a simple ownership-safe default allocator, not a pooling policy.
 * Applications can provide a different {@link TensorAllocator} to use pooled,
 * pinned, mapped, or budget-enforcing storage.</p>
 */
public final class ArenaTensorAllocator implements TensorAllocator {

    /** Shared stateless allocator instance for ordinary host allocations. */
    public static final ArenaTensorAllocator INSTANCE = new ArenaTensorAllocator();

    @Override
    public TensorStorage allocate(long sizeBytes) {
        return allocate(sizeBytes, 1L);
    }

    @Override
    public ArenaTensorStorage allocate(long sizeBytes, long alignmentBytes) {
        if (sizeBytes < 0L) {
            throw new IllegalArgumentException("sizeBytes must be non-negative");
        }
        if (alignmentBytes <= 0L || (alignmentBytes & (alignmentBytes - 1L)) != 0L) {
            throw new IllegalArgumentException("alignmentBytes must be a positive power of two");
        }

        Arena arena = Arena.ofShared();
        try {
            MemorySegment segment = arena.allocate(sizeBytes, alignmentBytes);
            return new ArenaTensorStorage(arena, segment);
        } catch (RuntimeException | Error failure) {
            try {
                arena.close();
            } catch (RuntimeException | Error closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }
}
