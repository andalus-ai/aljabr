/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.loader;

import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Bounded LRU cache for memory-mapped SafeTensors resources.
 *
 * <p>The cache owns one session reference per entry. Callers that need tensor
 * memory to survive eviction, invalidation, or application shutdown must use
 * {@link #acquire(Path, SafetensorFFMLoader)} and close the returned lease.
 * Removing an entry retires only the cache's owner reference; mapped memory is
 * released after the final caller lease is closed.
 *
 * <p>{@link #getOrLoad(Path, SafetensorFFMLoader)} remains for compatibility.
 * Its bare result is not protected against concurrent eviction and should not
 * be used by new code.
 */
@ApplicationScoped
public class SafetensorLoadCache {

    private static final Logger log = Logger.getLogger(SafetensorLoadCache.class);

    @Inject
    SafetensorLoaderConfig config;

    @Inject
    SafetensorMetrics metrics;

    private LinkedHashMap<Path, CacheEntry> cache;

    private void ensureInitialized() {
        if (cache == null) {
            cache = new LinkedHashMap<>(16, 0.75f, true);
        }
    }

    /**
     * Acquire a caller-owned lease to a cached or newly loaded single-file result.
     * The lease keeps tensor memory valid even after cache eviction or shutdown.
     */
    public LoadLease acquire(Path path, SafetensorFFMLoader loader) {
        if (!config.cache().enabled()) {
            SafetensorLoadResult result = loader.load(path);
            CacheEntry uncached = new CacheEntry(result, Instant.now());
            try {
                LoadLease lease = uncached.acquire();
                uncached.retire();
                return lease;
            } catch (RuntimeException | Error failure) {
                uncached.retire();
                throw failure;
            }
        }

        List<CacheEntry> retired = new ArrayList<>();
        LoadLease acquired;
        try {
            synchronized (this) {
                ensureInitialized();
                CacheEntry entry = cache.get(path);
                if (entry != null && !entry.result().isClosed() && !isExpired(entry)) {
                    acquired = entry.acquire();
                    log.debugf("SafetensorLoadCache: HIT [%s]", path.getFileName());
                } else {
                    if (entry != null) {
                        cache.remove(path);
                        retired.add(entry);
                    }
                    SafetensorLoadResult result = loader.load(path);
                    entry = new CacheEntry(result, Instant.now());
                    cache.put(path, entry);
                    acquired = entry.acquire();
                    evictOverflow(retired);
                    log.debugf("SafetensorLoadCache: MISS [%s]", path.getFileName());
                }
            }
        } finally {
            retired.forEach(CacheEntry::retire);
        }
        return acquired;
    }

    /**
     * Compatibility API returning a cache-owned result without a caller lease.
     *
     * @deprecated Use {@link #acquire(Path, SafetensorFFMLoader)} and close the
     * returned lease. This method cannot protect a caller from concurrent eviction.
     */
    @Deprecated
    public synchronized SafetensorLoadResult getOrLoad(Path path, SafetensorFFMLoader loader) {
        if (!config.cache().enabled()) {
            return loader.load(path);
        }
        ensureInitialized();
        CacheEntry entry = cache.get(path);
        if (entry != null && !entry.result().isClosed() && !isExpired(entry)) {
            return entry.result();
        }
        if (entry != null) {
            cache.remove(path);
            entry.retire();
        }
        SafetensorLoadResult result = loader.load(path);
        cache.put(path, new CacheEntry(result, Instant.now()));
        List<CacheEntry> retired = new ArrayList<>();
        evictOverflow(retired);
        retired.forEach(CacheEntry::retire);
        return result;
    }

    public void invalidate(Path path) {
        CacheEntry retired;
        synchronized (this) {
            retired = cache == null ? null : cache.remove(path);
        }
        if (retired != null) {
            log.infof("SafetensorLoadCache: invalidated [%s]", path.getFileName());
            retired.retire();
        }
    }

    /** Retire all entries without invalidating resources held by active leases. */
    public void invalidateAll() {
        List<CacheEntry> retired;
        synchronized (this) {
            if (cache == null || cache.isEmpty()) {
                return;
            }
            retired = new ArrayList<>(cache.values());
            cache.clear();
        }
        log.infof("SafetensorLoadCache: invalidating all %d entries", retired.size());
        retired.forEach(CacheEntry::retire);
    }

    public synchronized int size() {
        return cache == null ? 0 : cache.size();
    }

    @PreDestroy
    void onShutdown() {
        log.info("SafetensorLoadCache: application shutdown — retiring cached files");
        invalidateAll();
    }

    private void evictOverflow(List<CacheEntry> retired) {
        int maxSize = Math.max(1, config.cache().maxSize());
        Iterator<Map.Entry<Path, CacheEntry>> iterator = cache.entrySet().iterator();
        while (cache.size() > maxSize && iterator.hasNext()) {
            Map.Entry<Path, CacheEntry> eldest = iterator.next();
            iterator.remove();
            retired.add(eldest.getValue());
            log.debugf("SafetensorLoadCache: evicting [%s] (LRU, size limit=%d)",
                    eldest.getKey().getFileName(), maxSize);
        }
    }

    private boolean isExpired(CacheEntry entry) {
        long ttlSeconds = config.cache().ttlSeconds();
        return ttlSeconds > 0 && Instant.now().isAfter(entry.insertedAt().plusSeconds(ttlSeconds));
    }

    private void closeResultQuietly(SafetensorLoadResult result) {
        try {
            if (result != null && !result.isClosed()) {
                result.close();
                metrics.recordClose();
            }
        } catch (RuntimeException failure) {
            log.warnf(failure, "SafetensorLoadCache: error closing uncached result");
        }
    }

    private final class CacheEntry {
        private final SafetensorLoadResult result;
        private final SafetensorShardSession session;
        private final Instant insertedAt;
        private final AtomicBoolean closeMetricRecorded = new AtomicBoolean();

        private CacheEntry(SafetensorLoadResult result, Instant insertedAt) {
            this.result = result;
            this.insertedAt = insertedAt;
            this.session = SafetensorShardSession.singleFile(result);
        }

        private SafetensorLoadResult result() { return result; }
        private Instant insertedAt() { return insertedAt; }

        private LoadLease acquire() {
            return new LoadLease(this, session.retain());
        }

        private void retire() {
            session.close();
            recordCloseIfReleased();
        }

        private void release(SafetensorShardSession.SessionLease lease) {
            lease.close();
            recordCloseIfReleased();
        }

        private void recordCloseIfReleased() {
            if (session.resourcesClosed() && closeMetricRecorded.compareAndSet(false, true)) {
                metrics.recordClose();
            }
        }
    }

    /** A caller-owned reference to the result's mapped tensor memory. */
    public final class LoadLease implements AutoCloseable {
        private final CacheEntry entry;
        private final SafetensorShardSession.SessionLease lease;
        private final AtomicBoolean closed = new AtomicBoolean();

        private LoadLease(CacheEntry entry, SafetensorShardSession.SessionLease lease) {
            this.entry = entry;
            this.lease = lease;
        }

        /** Returns the live result while this lease remains open. */
        public SafetensorLoadResult result() {
            if (closed.get()) {
                throw new IllegalStateException("SafeTensors load lease is closed.");
            }
            if (entry.result().isClosed()) {
                throw new IllegalStateException("SafeTensors load result is closed.");
            }
            return entry.result();
        }

        public boolean isClosed() {
            return closed.get();
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                entry.release(lease);
            }
        }
    }
}
