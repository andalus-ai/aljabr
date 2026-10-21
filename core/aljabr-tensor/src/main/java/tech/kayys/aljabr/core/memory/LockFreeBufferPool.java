package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lock-free, size-bucketed off-heap memory pool (Treiber stacks per bucket).
 * This is a drop-in alternative to OffHeapBufferPool for high-concurrency
 * workloads.
 * @author bhangun
 */
public final class LockFreeBufferPool implements AutoCloseable {

    // Buckets: 64 B, 128 B, 256 B, … up to 64 MiB
    private static final int MIN_BUCKET_SHIFT = 6; // 2^6 = 64 bytes
    private static final int MAX_BUCKET_SHIFT = 26; // 2^26 = 64 MiB
    static final long MAX_BUCKET_BYTES = 1L << MAX_BUCKET_SHIFT;
    private static final int NUM_BUCKETS = MAX_BUCKET_SHIFT - MIN_BUCKET_SHIFT + 1;

    /** Treiber stack node holding a MemorySegment. */
    private static final class Node {
        final MemorySegment seg;
        final Node next;

        Node(MemorySegment seg, Node next) {
            this.seg = seg;
            this.next = next;
        }
    }

    @SuppressWarnings("unchecked")
    private final AtomicReference<Node>[] heads = new AtomicReference[NUM_BUCKETS];
    private final Arena arena;

    // telemetry
    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder bytesPooled = new LongAdder();

    public LockFreeBufferPool() {
        this(Arena.ofShared());
    }

    public LockFreeBufferPool(Arena arena) {
        this.arena = arena;
        for (int i = 0; i < NUM_BUCKETS; i++)
            heads[i] = new AtomicReference<>();
    }

    public MemorySegment acquire(long byteSize) {
        int idx = bucketFor(byteSize);
        if (idx >= 0) {
            AtomicReference<Node> head = heads[idx];
            while (true) {
                Node h = head.get();
                if (h == null)
                    break;
                if (head.compareAndSet(h, h.next)) {
                    hits.increment();
                    return h.seg;
                }
            }
        }
        misses.increment();
        long allocSize = (idx >= 0) ? bucketSize(idx) : byteSize;
        return arena.allocate(allocSize);
    }

    public void release(MemorySegment seg) {
        int idx = bucketFor(seg.byteSize());
        if (idx >= 0) {
            AtomicReference<Node> head = heads[idx];
            Node n;
            do {
                n = new Node(seg, head.get());
            } while (!head.compareAndSet(n.next, n));
            bytesPooled.add(seg.byteSize());
        }
        // Oversized segments are retained by the arena and freed on close()
    }

    public Arena arena() {
        return arena;
    }

    public long hits() {
        return hits.sum();
    }

    public long misses() {
        return misses.sum();
    }

    public long bytesPooled() {
        return bytesPooled.sum();
    }

    public String stats() {
        long total = hits() + misses();
        double hitRate = total == 0 ? 0.0 : 100.0 * hits() / total;
        return String.format("LockFreeBufferPool[hits=%d, misses=%d, hitRate=%.1f%%, pooled=%s]",
                hits(), misses(), hitRate, humanBytes(bytesPooled()));
    }

    @Override
    public void close() {
        arena.close();
    }

    private static int bucketFor(long byteSize) {
        if (byteSize > MAX_BUCKET_BYTES || byteSize <= 0)
            return -1;
        int shift = 64 - Long.numberOfLeadingZeros(byteSize - 1);
        int adjusted = Math.max(shift, MIN_BUCKET_SHIFT);
        return adjusted - MIN_BUCKET_SHIFT;
    }

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
