package tech.kayys.aljabr.core.memory;

/**
 * Config-driven factory that selects between the RocksDB and HelixDB
 * {@code UnifiedMemoryStore} implementations at runtime.
 *
 * <p>
 * The implementation is loaded via {@link java.util.ServiceLoader} so that
 * neither {@code aljabr-rocksdb} nor {@code aljabr-helixdb} need to
 * be on the
 * compile-time classpath of the core module. The appropriate JAR just needs to
 * be present at runtime.
 *
 * <h2>Configuration</h2>
 * <p>
 * Select the backend by setting the system property or environment variable:
 * 
 * <pre>
 *   # JVM system property (takes precedence)
 *   -Daljabr.memory.store=rocksdb   (default)
 *   -Daljabr.memory.store=helixdb
 *
 *   # Environment variable (fallback)
 *   ALJABR_MEMORY_STORE=helixdb
 * </pre>
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * MemoryStoreConfig config = MemoryStoreConfig.builder()
 *         .dataDir("/var/aljabr/kv")
 *         .maxCacheMiB(512)
 *         .build();
 *
 * try (var store = MemoryStoreFactory.open(config)) {
 *     store.put("kv-key".getBytes(), tensorBytes);
 *     Optional<MemorySegment> seg = store.getZeroCopy("kv-key".getBytes(), arena);
 * }
 * }</pre>
 * @author bhangun
 */
public final class MemoryStoreFactory {

    /** System property key for backend selection. */
    public static final String PROPERTY_KEY = "aljabr.memory.store";
    public static final String ENV_KEY = "ALJABR_MEMORY_STORE";

    /** Backend identifier for RocksDB. */
    public static final String BACKEND_ROCKSDB = "rocksdb";
    public static final String BACKEND_HELIXDB = "helixdb";

    private MemoryStoreFactory() {
    }

    /**
     * Returns the configured backend name ({@code "rocksdb"} or {@code "helixdb"}).
     *
     * <p>
     * Resolution order:
     * <ol>
     * <li>System property {@value #PROPERTY_KEY}</li>
     * <li>Environment variable {@value #ENV_KEY}</li>
     * <li>Default: {@value #BACKEND_ROCKSDB}</li>
     * </ol>
     *
     * @return the resolved backend identifier (lower-cased, trimmed)
     */
    public static String resolvedBackend() {
        String prop = System.getProperty(PROPERTY_KEY);
        if (prop != null && !prop.isBlank())
            return prop.trim().toLowerCase();
        String env = System.getenv(ENV_KEY);
        if (env != null && !env.isBlank())
            return env.trim().toLowerCase();
        return BACKEND_ROCKSDB;
    }

    /**
     * Returns {@code true} if the resolved backend is HelixDB.
     *
     * @return whether HelixDB is selected
     */
    public static boolean isHelixDb() {
        return BACKEND_HELIXDB.equals(resolvedBackend());
    }

    /**
     * Returns {@code true} if the resolved backend is RocksDB (the default).
     *
     * @return whether RocksDB is selected
     */
    public static boolean isRocksDb() {
        return BACKEND_ROCKSDB.equals(resolvedBackend());
    }

    /**
     * Immutable configuration for opening a memory store.
     *
     * <p>
     * Construct with the builder:
     * 
     * <pre>{@code
     * MemoryStoreConfig cfg = MemoryStoreConfig.builder()
     *         .dataDir("/var/aljabr")
     *         .maxCacheMiB(256)
     *         .build();
     * }</pre>
     */
    public record MemoryStoreConfig(
            String dataDir,
            long maxCacheMiB,
            boolean readOnly,
            int maxOpenFiles) {
        /** Default data directory if not explicitly set. */
        private static final String DEFAULT_DATA_DIR = System.getProperty("user.home") + "/.aljabr/kv";
        private static final long DEFAULT_CACHE_MIB = 256;
        private static final int DEFAULT_OPEN_FILES = 512;

        public static Builder builder() {
            return new Builder();
        }

        /** Fluent builder for {@link MemoryStoreConfig}. */
        public static final class Builder {
            private String dataDir = DEFAULT_DATA_DIR;
            private long maxCacheMiB = DEFAULT_CACHE_MIB;
            private boolean readOnly = false;
            private int maxOpenFiles = DEFAULT_OPEN_FILES;

            public Builder dataDir(String dir) {
                this.dataDir = dir;
                return this;
            }

            public Builder maxCacheMiB(long mib) {
                this.maxCacheMiB = mib;
                return this;
            }

            public Builder readOnly(boolean ro) {
                this.readOnly = ro;
                return this;
            }

            public Builder maxOpenFiles(int n) {
                this.maxOpenFiles = n;
                return this;
            }

            public MemoryStoreConfig build() {
                return new MemoryStoreConfig(dataDir, maxCacheMiB, readOnly, maxOpenFiles);
            }
        }
    }
}
