package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class GgufPreparedCachePolicyTest {
    private static final String Q4_MIN_ROWS = "aljabr.gguf.q4k.cache_min_rows";
    private static final String Q4_MAX_BYTES = "aljabr.gguf.q4k.cache_max_bytes";

    @Test
    void resolvesPreparedCacheFamilyByTypeId() {
        assertFamily(GgufPreparedCachePolicy.Family.Q32, GgmlType.Q4_0, GgmlType.Q4_1, GgmlType.Q5_0, GgmlType.Q5_1);
        assertFamily(GgufPreparedCachePolicy.Family.Q2K, GgmlType.Q2_K);
        assertFamily(GgufPreparedCachePolicy.Family.Q3K, GgmlType.Q3_K);
        assertFamily(GgufPreparedCachePolicy.Family.Q4K, GgmlType.Q4_K);
        assertFamily(GgufPreparedCachePolicy.Family.Q5K, GgmlType.Q5_K);
        assertFamily(GgufPreparedCachePolicy.Family.Q6K, GgmlType.Q6_K);
        assertFamily(
                GgufPreparedCachePolicy.Family.Q8,
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

        assertNull(GgufPreparedCachePolicy.preparedMatrixCacheFamily(GgmlType.F32.id));
        assertNull(GgufPreparedCachePolicy.preparedMatrixCacheFamily(GgmlType.I8.id));
        assertNull(GgufPreparedCachePolicy.preparedMatrixCacheFamily(-1));
        assertNull(GgufPreparedCachePolicy.preparedMatrixCacheFamily(10_000));
    }

    @Test
    void refreshesCachedFamilyPolicyWhenPropertiesChange() {
        String previousMinRows = System.getProperty(Q4_MIN_ROWS);
        String previousMaxBytes = System.getProperty(Q4_MAX_BYTES);
        try {
            System.setProperty(Q4_MIN_ROWS, "4");
            System.setProperty(Q4_MAX_BYTES, "1k");
            GgufPreparedCachePolicy.CachePolicy first = GgufPreparedCachePolicy.Family.Q4K.policy();

            assertEquals(4, first.minRows());
            assertEquals(1024L, first.maxBytes());

            System.setProperty(Q4_MIN_ROWS, "2");
            System.setProperty(Q4_MAX_BYTES, "2k");
            GgufPreparedCachePolicy.CachePolicy second = GgufPreparedCachePolicy.Family.Q4K.policy();

            assertEquals(2, second.minRows());
            assertEquals(2L * 1024L, second.maxBytes());
        } finally {
            restoreProperty(Q4_MIN_ROWS, previousMinRows);
            restoreProperty(Q4_MAX_BYTES, previousMaxBytes);
        }
    }

    @Test
    void admissionPolicySkipsFullBudgetPolicyBelowRowFloor() {
        String previousMinRows = System.getProperty(Q4_MIN_ROWS);
        String previousMaxBytes = System.getProperty(Q4_MAX_BYTES);
        try {
            System.setProperty(Q4_MIN_ROWS, "4");
            System.setProperty(Q4_MAX_BYTES, "1k");

            assertNull(GgufPreparedCachePolicy.Family.Q4K.admissionPolicy(3));

            System.setProperty(Q4_MIN_ROWS, "2");
            System.setProperty(Q4_MAX_BYTES, "2k");
            GgufPreparedCachePolicy.CachePolicy admitted = GgufPreparedCachePolicy.Family.Q4K.admissionPolicy(2);

            assertEquals(2, admitted.minRows());
            assertEquals(2L * 1024L, admitted.maxBytes());
        } finally {
            restoreProperty(Q4_MIN_ROWS, previousMinRows);
            restoreProperty(Q4_MAX_BYTES, previousMaxBytes);
        }
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }

    private static void assertFamily(GgufPreparedCachePolicy.Family family, GgmlType... types) {
        for (GgmlType type : types) {
            assertEquals(family, GgufPreparedCachePolicy.preparedMatrixCacheFamily(type.id));
        }
    }
}
