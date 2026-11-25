package tech.kayys.aljabr.core.memory.helixdb;

import tech.kayys.aljabr.core.memory.offheap.OffHeapMemoryStore;

/**
 * @deprecated This historical HelixDB name is an ephemeral in-process FFM
 * store, not a HelixDB integration. Use {@link OffHeapMemoryStore}.
 */
@Deprecated(forRemoval = true)
public final class HelixDbMemoryStore extends OffHeapMemoryStore {
    public HelixDbMemoryStore(String ignoredPath) { super(ignoredPath); }
    public HelixDbMemoryStore(String ignoredPath, Config config) {
        super(ignoredPath, new OffHeapMemoryStore.Config(config.maxStorageMiB()));
    }
    /** @deprecated use {@link OffHeapMemoryStore.Config}. */
    @Deprecated(forRemoval = true)
    public static final class Config {
        private final long maxStorageMiB;
        public Config(long maxStorageMiB) { this.maxStorageMiB = maxStorageMiB; }
        public long maxStorageMiB() { return maxStorageMiB; }
        public static Config defaults() { return new Config(512L); }
        public static Config forTest() { return new Config(32L); }
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private long maxStorageMiB = 512L;
            public Builder maxStorageMiB(long value) { this.maxStorageMiB = value; return this; }
            public Config build() { return new Config(maxStorageMiB); }
        }
    }
}
