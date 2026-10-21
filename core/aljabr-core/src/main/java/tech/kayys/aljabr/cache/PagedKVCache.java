package tech.kayys.aljabr.cache;

import java.nio.ByteBuffer;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Minimal PagedKVCache implementation relocated into aljabr namespace.
 * Behavior: store pages as direct ByteBuffer of floats, evict oldest entries
 * when capacity exceeded
 * and notify eviction listeners.
 * @author bhangun
 */
public class PagedKVCache {
    private final long capacityBytes;
    private long currentBytes = 0;
    // insertion-order map to evict oldest entries
    private final LinkedHashMap<Long, ByteBuffer> pages = new LinkedHashMap<>();
    private final CopyOnWriteArrayList<Consumer<Long>> evictionListeners = new CopyOnWriteArrayList<>();

    public PagedKVCache(long capacityBytes) {
        this.capacityBytes = capacityBytes;
    }

    public synchronized void put(long key, float[] data) {
        // create direct byte buffer and fill with floats
        int bytes = data.length * 4;
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes).order(java.nio.ByteOrder.nativeOrder());
        buf.asFloatBuffer().put(data);
        // position at zero for consumers
        buf.position(0);

        // if replacing existing page, remove old size
        ByteBuffer prev = pages.put(key, buf);
        if (prev != null) {
            currentBytes -= prev.capacity();
        }
        currentBytes += buf.capacity();
        evictIfNeeded();
    }

    public synchronized ByteBuffer getView(long key) {
        ByteBuffer b = pages.get(key);
        if (b == null)
            return null;
        return b.duplicate().position(0);
    }

    public synchronized ByteBuffer pin(long key) {
        // For this minimal implementation, pin simply returns a duplicate of the stored
        // buffer
        return getView(key);
    }

    public synchronized void unpin(long key) {
        // no-op for minimal implementation
    }

    public void addEvictionListener(Consumer<Long> listener) {
        evictionListeners.add(listener);
    }

    private void evictIfNeeded() {
        if (capacityBytes <= 0)
            return;
        Iterator<Map.Entry<Long, ByteBuffer>> it = pages.entrySet().iterator();
        while (currentBytes > capacityBytes && it.hasNext()) {
            Map.Entry<Long, ByteBuffer> e = it.next();
            long k = e.getKey();
            ByteBuffer v = e.getValue();
            it.remove();
            currentBytes -= v.capacity();
            // notify listeners asynchronously (but fire in calling thread for simplicity)
            for (Consumer<Long> l : evictionListeners) {
                try {
                    l.accept(k);
                } catch (Throwable ignored) {
                }
            }
        }
    }
}
