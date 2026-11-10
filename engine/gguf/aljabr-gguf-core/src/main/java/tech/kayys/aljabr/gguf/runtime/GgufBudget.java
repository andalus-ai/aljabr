package tech.kayys.aljabr.gguf.runtime;

import java.util.Locale;

/**
 * Memory-budget helpers for GGUF prepared caches and runtime probes.
 *
 * <p>Prepared matrices trade memory for faster repeated mat-vec calls. This
 * class keeps the default budget heap-aware and makes every related property
 * accept the same human-readable byte suffixes.</p>
 * @author bhangun
 */
public final class GgufBudget {
    private static final long KIB = 1024L;
    private static final long MIB = KIB * 1024L;
    private static final long GIB = MIB * 1024L;
    private static final long DEFAULT_MIN_BYTES = 512L * MIB;
    private static final long DEFAULT_MAX_BYTES = 2L * GIB;
    private static final int DEFAULT_HEAP_DIVISOR = 4;
    private static final long DEFAULT_PREPARED_CACHE_BYTES =
            boundedHeapBudget(DEFAULT_MIN_BYTES, DEFAULT_MAX_BYTES, DEFAULT_HEAP_DIVISOR);

    private GgufBudget() {
    }

    public static long defaultPreparedCacheBytes() {
        return DEFAULT_PREPARED_CACHE_BYTES;
    }

    public static long defaultAutoPrepareBytes() {
        return defaultPreparedCacheBytes();
    }

    public static long byteSizeProperty(String property, long defaultValue) {
        String configured = System.getProperty(property);
        if (configured == null || configured.isBlank()) {
            return Math.max(0L, defaultValue);
        }
        try {
            return Math.max(0L, parseByteSize(configured));
        } catch (RuntimeException ignored) {
            return Math.max(0L, defaultValue);
        }
    }

    public static long parseByteSize(String value) {
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (normalized.isEmpty()) {
            throw new NumberFormatException("empty byte size");
        }
        long multiplier = 1L;
        String digits = normalized;
        if (normalized.endsWith("kib")) {
            multiplier = KIB;
            digits = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("kb")) {
            multiplier = KIB;
            digits = normalized.substring(0, normalized.length() - 2);
        } else if (normalized.endsWith("k")) {
            multiplier = KIB;
            digits = normalized.substring(0, normalized.length() - 1);
        } else if (normalized.endsWith("mib")) {
            multiplier = MIB;
            digits = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("mb")) {
            multiplier = MIB;
            digits = normalized.substring(0, normalized.length() - 2);
        } else if (normalized.endsWith("m")) {
            multiplier = MIB;
            digits = normalized.substring(0, normalized.length() - 1);
        } else if (normalized.endsWith("gib")) {
            multiplier = GIB;
            digits = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("gb")) {
            multiplier = GIB;
            digits = normalized.substring(0, normalized.length() - 2);
        } else if (normalized.endsWith("g")) {
            multiplier = GIB;
            digits = normalized.substring(0, normalized.length() - 1);
        } else if (normalized.endsWith("b")) {
            digits = normalized.substring(0, normalized.length() - 1);
        }
        return Math.multiplyExact(Long.parseLong(digits.trim()), multiplier);
    }

    private static long boundedHeapBudget(long minBytes, long maxBytes, int heapDivisor) {
        long maxHeap = Runtime.getRuntime().maxMemory();
        long heapBudget = maxHeap <= 0L ? minBytes : maxHeap / Math.max(1, heapDivisor);
        return Math.max(minBytes, Math.min(maxBytes, heapBudget));
    }
}
