package tech.kayys.aljabr.core.memory;

/**
 * Explicit factory for optional {@link UnifiedMemoryStore} providers.
 *
 * <p>No storage provider is selected implicitly. Add the corresponding provider
 * module to the runtime classpath and either call its explicit factory method or
 * configure {@value #PROPERTY_KEY}/{@value #ENV_KEY} before calling {@link #open(String)}.
 *
 * <p>RocksDB is a durable key-value adapter. The off-heap provider is an
 * in-process FFM implementation and does not persist data across JVM restarts.
 * The HelixDB-named provider is a deprecated compatibility adapter.
 */
public final class MemoryStoreFactory {

    public static final String PROPERTY_KEY = "aljabr.memory.store";
    public static final String ENV_KEY = "ALJABR_MEMORY_STORE";

    public static final String BACKEND_ROCKSDB = "rocksdb";
    public static final String BACKEND_OFFHEAP = "offheap";
    /** @deprecated use {@link #BACKEND_OFFHEAP}. */
    @Deprecated(forRemoval = true)
    public static final String BACKEND_HELIXDB = "helixdb";

    private static final String ROCKSDB_CLASS =
            "tech.kayys.aljabr.core.memory.rocksdb.RocksDbMemoryStore";
    private static final String OFFHEAP_CLASS =
            "tech.kayys.aljabr.core.memory.offheap.OffHeapMemoryStore";
    private static final String HELIXDB_CLASS =
            "tech.kayys.aljabr.core.memory.helixdb.HelixDbMemoryStore";

    private MemoryStoreFactory() {
    }

    /**
     * Opens the explicitly configured provider.
     *
     * @throws IllegalStateException if no provider is configured
     * @throws IllegalArgumentException if the configured provider is unknown
     */
    public static UnifiedMemoryStore open(String dbPath) {
        return switch (resolvedBackend()) {
            case BACKEND_ROCKSDB -> openRocksDb(dbPath);
            case BACKEND_OFFHEAP -> openOffHeap();
            case BACKEND_HELIXDB -> openHelixDb(dbPath);
            default -> throw new IllegalArgumentException(
                    "Unsupported memory store '" + resolvedBackend()
                            + "'. Supported providers: " + BACKEND_ROCKSDB + ", " + BACKEND_OFFHEAP + ", " + BACKEND_HELIXDB);
        };
    }

    /** Opens the optional RocksDB durable-storage provider. */
    public static UnifiedMemoryStore openRocksDb(String dbPath) {
        return instantiate(ROCKSDB_CLASS, dbPath, "aljabr-rocksdb");
    }

    /** Opens the canonical in-process FFM off-heap provider. */
    public static UnifiedMemoryStore openOffHeap() {
        return instantiateWithoutPath(OFFHEAP_CLASS, "aljabr-offheap");
    }

    /** @deprecated the in-process provider does not use a filesystem path. */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore openOffHeap(String ignoredPath) {
        return openOffHeap();
    }

    /**
     * Opens the deprecated compatibility provider.
     *
     * <p>The historical HelixDB name is retained temporarily for source
     * compatibility; this implementation is not a persistent HelixDB adapter.
     */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore openHelixDb(String dbPath) {
        return instantiate(HELIXDB_CLASS, dbPath, "aljabr-helixdb");
    }

    /** @deprecated use {@link #openRocksDb(String)} */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore createRocksDb(String dbPath) {
        return openRocksDb(dbPath);
    }

    /** @deprecated use {@link #openOffHeap(String)} */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore createHelixDb(String dbPath) {
        return openHelixDb(dbPath);
    }

    /** @deprecated configure a provider explicitly and use {@link #open(String)} */
    @Deprecated(forRemoval = true)
    public static UnifiedMemoryStore createDefault(String dbPath) {
        return open(dbPath);
    }

    /**
     * Returns the explicitly configured provider name.
     *
     * @throws IllegalStateException when neither the system property nor
     *         environment variable selects a provider
     */
    public static String resolvedBackend() {
        String property = System.getProperty(PROPERTY_KEY);
        if (property != null && !property.isBlank()) {
            return property.trim().toLowerCase(java.util.Locale.ROOT);
        }
        String environment = System.getenv(ENV_KEY);
        if (environment != null && !environment.isBlank()) {
            return environment.trim().toLowerCase(java.util.Locale.ROOT);
        }
        throw new IllegalStateException(
                "No memory store provider configured. Set -D" + PROPERTY_KEY
                        + "=rocksdb|offheap (or legacy helixdb) or " + ENV_KEY
                        + ", or call an explicit provider factory.");
    }

    public static boolean isOffHeap() {
        return BACKEND_OFFHEAP.equals(configuredBackend());
    }

    /** @deprecated use {@link #isOffHeap()}. */
    @Deprecated(forRemoval = true)
    public static boolean isHelixDb() {
        return BACKEND_HELIXDB.equals(configuredBackend());
    }

    public static boolean isRocksDb() {
        return BACKEND_ROCKSDB.equals(configuredBackend());
    }

    private static String configuredBackend() {
        String property = System.getProperty(PROPERTY_KEY);
        if (property != null && !property.isBlank()) {
            return property.trim().toLowerCase(java.util.Locale.ROOT);
        }
        String environment = System.getenv(ENV_KEY);
        return environment == null || environment.isBlank()
                ? null
                : environment.trim().toLowerCase(java.util.Locale.ROOT);
    }

    private static UnifiedMemoryStore instantiateWithoutPath(String className, String module) {
        try {
            Class<?> providerClass = Class.forName(className);
            return (UnifiedMemoryStore) providerClass.getConstructor().newInstance();
        } catch (ClassNotFoundException exception) {
            throw new IllegalStateException(
                    "Provider class '" + className + "' is unavailable. Add the '"
                            + module + "' module to the runtime classpath.",
                    exception);
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(
                    "Failed to initialize memory store provider '" + module + "'.",
                    exception);
        }
    }

    private static UnifiedMemoryStore instantiate(String className, String dbPath, String module) {
        try {
            Class<?> providerClass = Class.forName(className);
            return (UnifiedMemoryStore) providerClass
                    .getConstructor(String.class)
                    .newInstance(dbPath);
        } catch (ClassNotFoundException exception) {
            throw new IllegalStateException(
                    "Provider class '" + className + "' is unavailable. Add the '"
                            + module + "' module to the runtime classpath.",
                    exception);
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(
                    "Failed to initialize memory store provider '" + module
                            + "' at " + dbPath,
                    exception);
        }
    }
}
