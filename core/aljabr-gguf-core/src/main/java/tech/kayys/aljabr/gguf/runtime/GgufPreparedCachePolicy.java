package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;

/**
 * Prepared-matrix cache policy and system-property parsing.
 *
 * <p>Keeping these knobs outside {@link GgufPreparedMatrixCache} leaves the
 * cache class focused on LRU storage while this class owns type-to-budget routing.</p>
 * @author bhangun
 */
final class GgufPreparedCachePolicy {
    private static final Family[] CACHE_FAMILIES = Family.values();
    private static final Family[] CACHE_FAMILIES_BY_TYPE_ID = cacheFamiliesByTypeId();

    private GgufPreparedCachePolicy() {
    }

    static boolean reservePreparedMatrixCacheBytes(
            Family family,
            long estimatedBytes,
            long[] reservedCacheBytes) {
        return reservePreparedMatrixCacheBytes(
                family,
                estimatedBytes,
                family == null ? 0L : family.maxBytes(),
                reservedCacheBytes);
    }

    static boolean reservePreparedMatrixCacheBytes(
            Family family,
            long estimatedBytes,
            long maxBytes,
            long[] reservedCacheBytes) {
        if (family == null) {
            return false;
        }
        int bucket = family.bucket();
        if (reservedCacheBytes[bucket] > maxBytes - estimatedBytes) {
            return false;
        }
        reservedCacheBytes[bucket] += estimatedBytes;
        return true;
    }

    static int preparedMatrixCacheBucketCount() {
        return CACHE_FAMILIES.length;
    }

    static int preparedMatrixCacheBucket(int typeId) {
        Family family = preparedMatrixCacheFamily(typeId);
        return family == null ? -1 : family.bucket();
    }

    static long preparedMatrixCacheMaxBytes(int typeId) {
        Family family = preparedMatrixCacheFamily(typeId);
        return family == null ? 0L : family.maxBytes();
    }

    static Family preparedMatrixCacheFamily(int typeId) {
        return typeId >= 0 && typeId < CACHE_FAMILIES_BY_TYPE_ID.length
                ? CACHE_FAMILIES_BY_TYPE_ID[typeId]
                : null;
    }

    private static int cacheMinRows(String property) {
        if (property == null || property.isBlank()) {
            return 32;
        }
        try {
            return Math.max(1, Integer.decode(property.trim()));
        } catch (RuntimeException ignored) {
            return 32;
        }
    }

    private static long cacheMaxBytes(String property) {
        if (property == null || property.isBlank()) {
            return GgufBudget.defaultPreparedCacheBytes();
        }
        try {
            return Math.max(0L, GgufBudget.parseByteSize(property));
        } catch (RuntimeException ignored) {
            return GgufBudget.defaultPreparedCacheBytes();
        }
    }

    record CachePolicy(int minRows, long maxBytes) {
    }

    enum Family {
        Q32(0, "aljabr.gguf.q32.cache_min_rows", "aljabr.gguf.q32.cache_max_bytes"),
        Q2K(1, "aljabr.gguf.q2k.cache_min_rows", "aljabr.gguf.q2k.cache_max_bytes"),
        Q3K(2, "aljabr.gguf.q3k.cache_min_rows", "aljabr.gguf.q3k.cache_max_bytes"),
        Q4K(3, "aljabr.gguf.q4k.cache_min_rows", "aljabr.gguf.q4k.cache_max_bytes"),
        Q5K(4, "aljabr.gguf.q5k.cache_min_rows", "aljabr.gguf.q5k.cache_max_bytes"),
        Q6K(5, "aljabr.gguf.q6k.cache_min_rows", "aljabr.gguf.q6k.cache_max_bytes"),
        Q8(6, "aljabr.gguf.q8.cache_min_rows", "aljabr.gguf.q8.cache_max_bytes");

        private final int bucket;
        private final String minRowsProperty;
        private final String maxBytesProperty;
        private volatile CachedMinRows cachedMinRows;
        private volatile CachedPolicy cachedPolicy;

        Family(int bucket, String minRowsProperty, String maxBytesProperty) {
            this.bucket = bucket;
            this.minRowsProperty = minRowsProperty;
            this.maxBytesProperty = maxBytesProperty;
        }

        int bucket() {
            return bucket;
        }

        int minRows() {
            return minRowsPolicy(System.getProperty(minRowsProperty)).value();
        }

        long maxBytes() {
            return policy().maxBytes();
        }

        CachePolicy admissionPolicy(int rowCount) {
            String minRowsValue = System.getProperty(minRowsProperty);
            CachedMinRows minRows = minRowsPolicy(minRowsValue);
            if (rowCount < minRows.value()) {
                return null;
            }
            return policy(minRowsValue, minRows.value());
        }

        CachePolicy policy() {
            String minRowsValue = System.getProperty(minRowsProperty);
            return policy(minRowsValue, minRowsPolicy(minRowsValue).value());
        }

        private CachePolicy policy(String minRowsValue, int minRows) {
            String maxBytesValue = System.getProperty(maxBytesProperty);
            CachedPolicy policy = cachedPolicy;
            if (policy != null && policy.matches(minRowsValue, maxBytesValue)) {
                return policy.values();
            }
            CachedPolicy updated = new CachedPolicy(
                    minRowsValue,
                    maxBytesValue,
                    new CachePolicy(minRows, cacheMaxBytes(maxBytesValue)));
            cachedPolicy = updated;
            return updated.values();
        }

        private CachedMinRows minRowsPolicy(String minRowsValue) {
            CachedMinRows minRows = cachedMinRows;
            if (minRows != null && minRows.matches(minRowsValue)) {
                return minRows;
            }
            CachedMinRows updated = new CachedMinRows(minRowsValue, cacheMinRows(minRowsValue));
            cachedMinRows = updated;
            return updated;
        }

    }

    private record CachedMinRows(String property, int value) {
        private boolean matches(String property) {
            return java.util.Objects.equals(this.property, property);
        }
    }

    private record CachedPolicy(String minRowsProperty, String maxBytesProperty, CachePolicy values) {
        private boolean matches(String minRowsProperty, String maxBytesProperty) {
            return java.util.Objects.equals(this.minRowsProperty, minRowsProperty)
                    && java.util.Objects.equals(this.maxBytesProperty, maxBytesProperty);
        }
    }

    private static Family[] cacheFamiliesByTypeId() {
        int maxTypeId = 0;
        for (GgmlType type : GgmlType.values()) {
            maxTypeId = Math.max(maxTypeId, type.id);
        }
        Family[] families = new Family[maxTypeId + 1];
        map(families, Family.Q32, GgmlType.Q4_0, GgmlType.Q4_1, GgmlType.Q5_0, GgmlType.Q5_1);
        map(families, Family.Q2K, GgmlType.Q2_K);
        map(families, Family.Q3K, GgmlType.Q3_K);
        map(families, Family.Q4K, GgmlType.Q4_K);
        map(families, Family.Q5K, GgmlType.Q5_K);
        map(families, Family.Q6K, GgmlType.Q6_K);
        map(families, Family.Q8,
                GgmlType.Q1_0,
                GgmlType.TQ1_0,
                GgmlType.TQ2_0,
                GgmlType.MXFP4,
                GgmlType.NVFP4,
                GgmlType.Q8_0,
                GgmlType.Q8_1,
                GgmlType.Q8_K,
                GgmlType.IQ4_NL,
                GgmlType.IQ4_XS);
        return families;
    }

    private static void map(Family[] families, Family family, GgmlType... types) {
        for (GgmlType type : types) {
            if (families[type.id] != null) {
                throw new IllegalStateException("Duplicate GGUF prepared cache family for type id: " + type.id);
            }
            families[type.id] = family;
        }
    }
}
