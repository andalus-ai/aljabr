package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Session-scoped, size-bucketed off-heap memory pool.
 * Normal acquire/release calls share a lifecycle read lock; close takes the
 * write lock and waits for in-flight pool operations before closing the arena.
 * Callers must stop using acquired segments before closing this pool.
 */
public final class OffHeapBufferPool implements AutoCloseable {
    private static final int MIN_BUCKET_SHIFT = 6;
    private static final int MAX_BUCKET_SHIFT = 26;
    static final long MAX_BUCKET_BYTES = 1L << MAX_BUCKET_SHIFT;
    private static final int NUM_BUCKETS = MAX_BUCKET_SHIFT - MIN_BUCKET_SHIFT + 1;

    @SuppressWarnings("unchecked")
    private final AtomicReference<Node>[] heads = new AtomicReference[NUM_BUCKETS];
    private final Arena arena;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    // Identity semantics are required: a lease is the exact segment instance returned by acquire().
    // Access is synchronized because multiple operations hold the lifecycle read lock concurrently.
    private final Map<MemorySegment, Boolean> activeSegments = new IdentityHashMap<>();
    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder bytesPooled = new LongAdder();

    private static final class Node {
        final MemorySegment segment;
        final Node next;
        Node(MemorySegment segment, Node next) { this.segment = segment; this.next = next; }
    }

    public OffHeapBufferPool() { this(Arena.ofShared()); }

    /** The supplied arena becomes owned by this pool and is closed by close(). */
    public OffHeapBufferPool(Arena arena) {
        this.arena = Objects.requireNonNull(arena, "arena must not be null");
        for (int i = 0; i < NUM_BUCKETS; i++) heads[i] = new AtomicReference<>();
    }

    /** Acquires at least byteSize bytes; zero is allowed, negative sizes are rejected. */
    public MemorySegment acquire(long byteSize) {
        if (byteSize < 0) throw new IllegalArgumentException("byteSize must be non-negative");
        var lock = lifecycle.readLock();
        lock.lock();
        try {
            ensureOpen();
            int idx = bucketFor(byteSize);
            if (idx >= 0) {
                AtomicReference<Node> head = heads[idx];
                while (true) {
                    Node current = head.get();
                    if (current == null) break;
                    if (head.compareAndSet(current, current.next)) {
                        hits.increment();
                        bytesPooled.add(-current.segment.byteSize());
                        markAcquired(current.segment);
                        return current.segment;
                    }
                }
            }
            misses.increment();
            MemorySegment segment = arena.allocate(idx >= 0 ? bucketSize(idx) : byteSize);
            markAcquired(segment);
            return segment;
        } finally { lock.unlock(); }
    }

    /** Returns a segment for reuse. The caller must not access it afterwards. */
    public void release(MemorySegment segment) {
        Objects.requireNonNull(segment, "segment must not be null");
        var lock = lifecycle.readLock();
        lock.lock();
        try {
            if (closed.get()) return;
            if (segment.scope() != arena.scope()) {
                throw new IllegalArgumentException("Segment must belong to this pool's arena");
            }
            synchronized (activeSegments) {
                if (activeSegments.remove(segment) == null) {
                    throw new IllegalArgumentException(
                            "Segment was not acquired from this pool or has already been released");
                }
            }
            int idx = bucketFor(segment.byteSize());
            if (idx >= 0) {
                AtomicReference<Node> head = heads[idx];
                Node node;
                do { node = new Node(segment, head.get()); }
                while (!head.compareAndSet(node.next, node));
                bytesPooled.add(segment.byteSize());
            }
        } finally { lock.unlock(); }
    }

    public Arena arena() { return arena; }
    public long hits() { return hits.sum(); }
    public long misses() { return misses.sum(); }
    public long bytesPooled() { return bytesPooled.sum(); }

    public String stats() {
        long hitCount = hits();
        long missCount = misses();
        long total = hitCount + missCount;
        double rate = total == 0 ? 0.0 : 100.0 * hitCount / total;
        return String.format("OffHeapBufferPool[hits=%d, misses=%d, hitRate=%.1f%%, pooled=%s]",
                hitCount, missCount, rate, humanBytes(bytesPooled()));
    }

    /** Waits for in-flight pool operations, then invalidates arena-backed segments. */
    @Override public void close() {
        var lock = lifecycle.writeLock();
        lock.lock();
        try {
            if (closed.compareAndSet(false, true)) {
                for (AtomicReference<Node> head : heads) head.set(null);
                synchronized (activeSegments) {
                    activeSegments.clear();
                }
                bytesPooled.reset();
                arena.close();
            }
        } finally { lock.unlock(); }
    }

    private void markAcquired(MemorySegment segment) {
        synchronized (activeSegments) {
            if (activeSegments.put(segment, Boolean.TRUE) != null) {
                throw new IllegalStateException("Segment is already checked out from this pool");
            }
        }
    }

    private void ensureOpen() {
        if (closed.get()) throw new IllegalStateException("OffHeapBufferPool is closed");
    }

    private static int bucketFor(long size) {
        if (size > MAX_BUCKET_BYTES || size <= 0) return -1;
        int shift = 64 - Long.numberOfLeadingZeros(size - 1);
        return Math.max(shift, MIN_BUCKET_SHIFT) - MIN_BUCKET_SHIFT;
    }

    private static long bucketSize(int idx) { return 1L << (idx + MIN_BUCKET_SHIFT); }

    private static String humanBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KiB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MiB", bytes / (1024.0 * 1024));
        return String.format("%.2f GiB", bytes / (1024.0 * 1024 * 1024));
    }
}
