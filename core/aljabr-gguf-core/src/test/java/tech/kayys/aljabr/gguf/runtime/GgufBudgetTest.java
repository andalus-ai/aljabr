package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufBudgetTest {
    private static final String BUDGET_PROPERTY = "aljabr.gguf.test_budget";

    @Test
    void parsesHumanReadableByteSizes() {
        assertEquals(12L, GgufBudget.parseByteSize("12"));
        assertEquals(1024L, GgufBudget.parseByteSize("1k"));
        assertEquals(2L * 1024L, GgufBudget.parseByteSize("2KB"));
        assertEquals(3L * 1024L * 1024L, GgufBudget.parseByteSize("3mib"));
        assertEquals(4L * 1024L * 1024L * 1024L, GgufBudget.parseByteSize("4g"));
    }

    @Test
    void readsSuffixBudgetPropertiesWithSafeFallback() {
        String previous = System.getProperty(BUDGET_PROPERTY);
        try {
            System.setProperty(BUDGET_PROPERTY, "1k");
            assertEquals(1024L, GgufBudget.byteSizeProperty(BUDGET_PROPERTY, 7L));

            System.setProperty(BUDGET_PROPERTY, "not-a-size");
            assertEquals(7L, GgufBudget.byteSizeProperty(BUDGET_PROPERTY, 7L));
        } finally {
            if (previous == null) {
                System.clearProperty(BUDGET_PROPERTY);
            } else {
                System.setProperty(BUDGET_PROPERTY, previous);
            }
        }
    }

    @Test
    void defaultPreparedCacheBudgetNeverDropsBelowLegacyFloor() {
        assertTrue(GgufBudget.defaultPreparedCacheBytes() >= 512L * 1024L * 1024L);
    }
}
