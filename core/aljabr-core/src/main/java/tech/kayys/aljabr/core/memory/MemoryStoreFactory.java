package tech.kayys.aljabr.core.memory;

/**
 * Config-driven factory for creating {@link UnifiedMemoryStore} instances.
 *
 * <h2>Backend selection</h2>
 * <p>
 * The backend is resolved in this order:
 * <ol>
 * <li>System property {@value #PROPERTY_KEY} (e.g.
 * {@code -Daljabr.memory.store=helixdb})</li>
 * <li>Environment variable {@value #ENV_KEY}</li>
 * <li>Default: {@value #BACKEND_ROCKSDB}</li>
 * </ol>
 *
 * <h2>ServiceLoader design</h2>
 * <p>
 * The store implementations ({@code aljabr-rocksdb},
 * {@code aljabr-helixdb})
 * are
 * loaded via reflection so that neither needs to be a compile-time dependency
 * of
 * {@code aljabr-core}. Simply include the desired JAR on the runtime
 * classpath.
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * // Auto-select via system property (-Daljabr.memory.store=helixdb):
 * try (UnifiedMemoryStore store = MemoryStoreFactory.open("/var/aljabr/kv")) {
 *     store.put(key, tensorSegment);
 *     store.getZeroCopy(key, expectedSize, arena).ifPresent(seg -> ...);
 * }
 *
 * // Explicit:
 * try (UnifiedMemoryStore store = MemoryStoreFactory.openHelixDb("/var/aljabr/kv")) {
 *     ...
 * }
 * }</pre>
 * @author bhangun
 */
public final class MemoryStoreFactory {

    /** System property key: {@code -Daljabr.memory.store=rocksdb|helixdb} */
    public static final String PROPERTY_KEY = "aljabr.memory.store";
    public static final String ENV_KEY = "ALJABR_MEMORY_STORE";

    public static final String BACKEND_ROCKSDB = "rocksdb";
    public static final String BACKEND_HELIXDB = "helixdb";

    private static final String ROCKSDB_CLASS = "tech.kayys.aljabr.core.memory.rocksdb.RocksDbMemoryStore";
    private static final String HELIXDB_CLASS = "tech.kayys.aljabr.core.memory.helixdb.HelixDbMemoryStore";

    private MemoryStoreFactory() {
    }

    // ── Auto-select ───────────────────────────────────────────────────────────

    /**
     * Opens a store using the backend selected by system property or environment
     * variable.
     *
     * @param dbPath directory path for the store's data files
     * @return the opened store
     */
    public static UnifiedMemoryStore open(String dbPath) {
        return switch (resolvedBackend()) {
            case BACKEND_HELIXDB -> openHelixDb(dbPath);
            default -> openRocksDb(dbPath);
        };
    }

    // ── Explicit factories ────────────────────────────────────────────────────

    /**
     * Creates a new RocksDB-backed store at {@code dbPath}.
     *
     * @param dbPath path where RocksDB files will be created/opened
     * @return the opened store
     * @throws RuntimeException if {@code aljabr-rocksdb} is not on the
     *                          classpath
     */
    public static UnifiedMemoryStore openRocksDb(String dbPath) {
        return instantiate(ROCKSDB_CLASS, dbPath, "aljabr-rocksdb");
    }

    /**
     * Creates a new HelixDB-backed store at {@code dbPath}.
     *
     * @param dbPath path where HelixDB files will be created/opened
     * @return the opened store
     * @throws RuntimeException if {@code aljabr-helixdb} is not on the
     *                          classpath
     */
    public static UnifiedMemoryStore openHelixDb(String dbPath) {
        return instantiate(HELIXDB_CLASS, dbPath, "aljabr-helixdb");
    }

    // ── Deprecated aliases (keep for backwards compat) ────────────────────────

    /** @deprecated use {@link #openRocksDb(String)} */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore createRocksDb(String dbPath) {
        return openRocksDb(dbPath);
    }

    /** @deprecated use {@link #openHelixDb(String)} */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore createHelixDb(String dbPath) {
        return openHelixDb(dbPath);
    }

    /** @deprecated use {@link #open(String)} */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore createDefault(String dbPath) {
        return open(dbPath);
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /**
     * Returns the name of the backend that {@link #open(String)} would select
     * given the current system property / environment variable configuration.
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

    /** {@code true} when {@link #open} would use HelixDB. */
    public static boolean isHelixDb() {
        return BACKEND_HELIXDB.equals(resolvedBackend());
    }

    /** {@code true} when {@link #open} would use RocksDB. */
    public static boolean isRocksDb() {
        return BACKEND_ROCKSDB.equals(resolvedBackend());
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static UnifiedMemoryStore instantiate(String className, String dbPath, String module) {
        try {
            Class<?> clazz = Class.forName(className);
            return (UnifiedMemoryStore) clazz.getConstructor(String.class).newInstance(dbPath);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(
                    "Backend class '" + className + "' not found. "
                            + "Add the '" + module + "' module to your runtime classpath.",
                    e);
        } catch (Exception e) {
            throw new RuntimeException("Failed to instantiate " + className + " at " + dbPath, e);
        }
    }
}
