package tech.kayys.aljabr.numerics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SplitMix64Test {
    @Test
    void equalSeedsProduceEqualSequences() {
        RandomSource first = new SplitMix64(42L);
        RandomSource second = new SplitMix64(42L);
        for (int i = 0; i < 100; i++) {
            assertEquals(first.nextLong(), second.nextLong());
        }
    }

    @Test
    void boundedIntegersStayWithinRange() {
        RandomSource random = new SplitMix64(7L);
        for (int i = 0; i < 10_000; i++) {
            int value = random.nextInt(13);
            assertTrue(value >= 0 && value < 13);
        }
        assertThrows(IllegalArgumentException.class, () -> random.nextInt(0));
    }

    @Test
    void doublesStayInUnitInterval() {
        RandomSource random = new SplitMix64(99L);
        for (int i = 0; i < 10_000; i++) {
            double value = random.nextDouble();
            assertTrue(value >= 0.0 && value < 1.0);
        }
    }

    @Test
    void splitCreatesReproducibleChildStreams() {
        RandomSource first = new SplitMix64(123L).split();
        RandomSource second = new SplitMix64(123L).split();
        for (int i = 0; i < 20; i++) {
            assertEquals(first.nextLong(), second.nextLong());
        }
    }
}
