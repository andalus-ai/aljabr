package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufSumTest {
    @Test
    void sumsFixedWidthVectorGroups() {
        float[] values = increasingValues(96);

        assertEquals(rangeSum(1, 16), GgufSum.sumFloatFixed(values, 0, 16), 0.0f);
        assertEquals(rangeSum(1, 16), GgufSum.sumFloat16(values, 0), 0.0f);
        assertEquals(rangeSum(33, 64), GgufSum.sumFloatFixed(values, 32, 32), 0.0f);
        assertEquals(rangeSum(33, 64), GgufSum.sumFloat32(values, 32), 0.0f);
    }

    @Test
    void fillsReusableGroupSumsForSixteenAndThirtyTwoWideGroups() {
        float[] values = increasingValues(96);
        GgufTensorOps.Q4KWorkBuffer workBuffer = new GgufTensorOps.Q4KWorkBuffer();

        float[] sums16 = GgufSum.vector16GroupSums(values, 48, workBuffer);
        assertEquals(rangeSum(1, 16), sums16[0], 0.0f);
        assertEquals(rangeSum(17, 32), sums16[1], 0.0f);
        assertEquals(rangeSum(33, 48), sums16[2], 0.0f);

        float[] sums32 = GgufSum.vector32GroupSums(values, 96, workBuffer);
        assertEquals(rangeSum(1, 32), sums32[0], 0.0f);
        assertEquals(rangeSum(33, 64), sums32[1], 0.0f);
        assertEquals(rangeSum(65, 96), sums32[2], 0.0f);
    }

    @Test
    void fillsReusableGroupSumsAcrossQuadPairAndSingleTails() {
        float[] values = increasingValues(224);
        GgufTensorOps.Q4KWorkBuffer workBuffer = new GgufTensorOps.Q4KWorkBuffer();

        float[] sums16 = GgufSum.vector16GroupSums(values, 112, workBuffer);
        assertRangeSums(sums16, 16, 7);

        float[] sums32 = GgufSum.vector32GroupSums(values, 224, workBuffer);
        assertRangeSums(sums32, 32, 7);
    }

    private static float[] increasingValues(int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = i + 1.0f;
        }
        return values;
    }

    private static void assertRangeSums(float[] sums, int groupSize, int groups) {
        for (int group = 0; group < groups; group++) {
            int first = group * groupSize + 1;
            int last = first + groupSize - 1;
            assertEquals(rangeSum(first, last), sums[group], 0.0f);
        }
    }

    private static float rangeSum(int firstInclusive, int lastInclusive) {
        return (firstInclusive + lastInclusive) * (lastInclusive - firstInclusive + 1) / 2.0f;
    }
}
