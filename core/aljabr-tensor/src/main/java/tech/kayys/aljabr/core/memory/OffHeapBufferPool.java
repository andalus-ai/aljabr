package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.atomic.LongAdder;

/**
 * A session-scoped, size-bucketed off-heap memory pool for zero-copy tensor
 * allocations.
 *
 * <h2>Design</h2>
 * <p>
 * ML inference allocates many intermediate tensors (activations, attention
 * scores, etc.)
 * at high frequency. Delegating these allocations to a JVM GC-managed heap
 * causes
 * significant GC pauses and cache thrashing.
 *
 * <p>
 * This pool maintains separate free-lists for each power-of-two size class
 * (from 64 bytes to {@value #MAX_BUCKET_BYTES} bytes). Requests that fall into
 * a known bucket are satisfied in O(1) without touching the OS. Requests larger
 * than {@value #MAX_BUCKET_BYTES} are allocated directly from the underlying
 * {@link Arena} without pooling (large tensors are typically not recycled
 * quickly).
 *
 * <h2>Lifecycle</h2>
 * 
 * <pre>{@code
 * try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
 *     CpuBackend backend = new CpuBackend(pool);
 *     Tensor result = backend.matmul(a, b);
 *     // use result …
 * } // ← ALL off-heap memory freed here in one syscall
 * }</pre>
 *
 * <h2>Thread safety</h2>
 * <p>
 * Each bucket's free-list is independently synchronized. The pool is safe
 * for concurrent use from multiple threads (e.g. parallel token generation).
 * @author bhangun
 */
public final class OffHeapBufferPool implements AutoCloseable {

    // Buckets: 64 B, 128 B, 256 B, … up to 64 MiB
    private static final int MIN_BUCKET_SHIFT = 6; // 2^6 = 64 bytes
    private static final int MAX_BUCKET_SHIFT = 26; // 2^26 = 64 MiB
    static final long MAX_BUCKET_BYTES = 1L << MAX_BUCKET_SHIFT;
    private static final int NUM_BUCKETS = MAX_BUCKET_SHIFT - MIN_BUCKET_SHIFT + 1;

    @SuppressWarnings("unchecked")
    private final java.util.concurrent.atomic.AtomicReference<Node>[] heads = new java.util.concurrent.atomic.AtomicReference[NUM_BUCKETS];

    private final Arena arena;
    private final java.util.concurrent.atomic.AtomicBoolean closed = new java.util.concurrent.atomic.AtomicBoolean();

    // Treiber node for lock-free stacks
    private static final class Node {
        final MemorySegment seg;
        final Node next;

        Node(MemorySegment seg, Node next) {
            this.seg = seg;
            this.next = next;
        }
    }

    // ── Telemetry ──────────────────────────────────────────────────────────────
    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder bytesPooled = new LongAdder();

    /** Creates a pool backed by a new shared (thread-safe) arena. */
    public OffHeapBufferPool() {
        this(Arena.ofShared());
    }

    /** Creates a pool backed by the supplied arena. */
    public OffHeapBufferPool(Arena arena) {
        this.arena = arena;
        for (int i = 0; i < NUM_BUCKETS; i++) {
            heads[i] = new java.util.concurrent.atomic.AtomicReference<>();
        }
    }

    // ── Core API ───────────────────────────────────────────────────────────────

    /**
     * Acquires a {@link MemorySegment} of at least {@code byteSize} bytes.
     *
     * <p>
     * If a suitable recycled segment exists in the matching bucket it is
     * returned immediately. Otherwise a fresh segment is allocated from the
     * underlying {@link Arena}.
     *
     * @param byteSize the minimum number of bytes needed
     * @return a segment whose {@link MemorySegment#byteSize()} is ≥
     *         {@code byteSize}
     */
    public MemorySegment acquire(long byteSize) {
        if (closed.get()) {
            throw new IllegalStateException("OffHeapBufferPool is closed");
        }
        int bucketIdx = bucketFor(byteSize);
        if (bucketIdx >= 0) {
            java.util.concurrent.atomic.AtomicReference<Node> head = heads[bucketIdx];
            while (true) {
                Node h = head.get();
                if (h == null)
                    break;
                if (head.compareAndSet(h, h.next)) {
                    hits.increment();
                    bytesPooled.add(-h.seg.byteSize());
                    return h.seg;
                }
            }
        }
        misses.increment();
        // Round up to the bucket boundary so the segment can be recycled later.
        long allocSize = (bucketIdx >= 0) ? bucketSize(bucketIdx) : byteSize;
        return arena.allocate(allocSize);
    }

    /**
     * Returns a segment to the pool for future reuse.
     *
     * <p>
     * The memory is <em>not</em> freed; it is held until {@link #close()} is
     * called.
     * Callers must not use the segment after calling this method.
     *
     * @param seg the segment to recycle
     */
    public void release(MemorySegment seg) {
        if (closed.get()) {
            return;
        }
        int bucketIdx = bucketFor(seg.byteSize());
        if (bucketIdx >= 0) {
            java.util.concurrent.atomic.AtomicReference<Node> head = heads[bucketIdx];
            Node n;
            do {
                n = new Node(seg, head.get());
            } while (!head.compareAndSet(n.next, n));
            bytesPooled.add(seg.byteSize());
        }
        // Segments larger than MAX_BUCKET_BYTES are not pooled — they will be freed
        // when
        // the arena closes.
    }

    /** Returns the underlying {@link Arena}. */
    public Arena arena() {
        return arena;
    }

    // ── Telemetry ──────────────────────────────────────────────────────────────

    /** Number of {@link #acquire} calls satisfied from the free-list. */
    public long hits() {
        return hits.sum();
    }

    /** Number of {@link #acquire} calls that required a fresh allocation. */
    public long misses() {
        return misses.sum();
    }

    /** Total bytes currently sitting in free-lists waiting to be recycled. */
    public long bytesPooled() {
        return bytesPooled.sum();
    }

    /**
     * Returns a human-readable summary for logging / diagnostics.
     *
     * @return pool statistics string
     */
    public String stats() {
        long total = hits() + misses();
        double hitRate = (total == 0) ? 0.0 : 100.0 * hits() / total;
        return String.format(
                "OffHeapBufferPool[hits=%d, misses=%d, hitRate=%.1f%%, pooled=%s]",
                hits(), misses(), hitRate, humanBytes(bytesPooled()));
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /**
     * Frees all native memory held by this pool in a single operation.
     *
     * <p>
     * After calling {@code close()} any segment previously acquired from
     * this pool becomes invalid and must not be accessed.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            for (java.util.concurrent.atomic.AtomicReference<Node> head : heads) {
                head.set(null);
            }
            bytesPooled.add(-bytesPooled.sum());
            arena.close();
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /**
     * Returns the bucket index for a given byte size, or {@code -1} if the
     * size exceeds {@link #MAX_BUCKET_BYTES} (i.e. no bucket).
     */
    private static int bucketFor(long byteSize) {
        if (byteSize > MAX_BUCKET_BYTES || byteSize <= 0)
            return -1;
        int shift = 64 - Long.numberOfLeadingZeros(byteSize - 1);
        int adjusted = Math.max(shift, MIN_BUCKET_SHIFT);
        return adjusted - MIN_BUCKET_SHIFT;
    }

    /** Returns the actual allocation size for a given bucket index. */
    private static long bucketSize(int idx) {
        return 1L << (idx + MIN_BUCKET_SHIFT);
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024)
            return bytes + " B";
        if (bytes < 1024 * 1024)
            return String.format("%.1f KiB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024)
            return String.format("%.1f MiB", bytes / (1024.0 * 1024));
        return String.format("%.2f GiB", bytes / (1024.0 * 1024 * 1024));
    }
}
