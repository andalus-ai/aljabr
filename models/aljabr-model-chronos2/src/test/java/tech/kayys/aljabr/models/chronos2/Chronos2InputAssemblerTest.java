package tech.kayys.aljabr.models.chronos2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class Chronos2InputAssemblerTest {
    private Chronos2Config config() {
        return new Chronos2Config(
                8, 4, 4, 4, 4, 8,
                8, 2, 16, 2, 2, 1e-6, 10000.0,
                false, false, List.of(0.1, 0.5, 0.9));
    }

    @Test
    void assemblesContextAndFutureFeaturesInTimeValueMaskOrder() {
        var input = new Chronos2InputAssembler(config()).assemble(
                new double[] {1, 2, 3, 4, 5},
                null,
                new double[] {6, Double.NaN},
                null,
                1);

        assertEquals(2, input.contextPatchCount());
        assertEquals(1, input.outputPatchCount());
        double[][] context = input.contextFeatures();
        assertEquals(12, context[0].length);
        assertEquals(-1.0, context[0][0], 1e-12);
        assertEquals(-5.0 / 8.0, context[0][3], 1e-12);
        assertEquals(-Math.sqrt(2.0), context[0][7], 1e-12);
        assertEquals(0.0, context[0][8], 0.0);
        assertEquals(1.0, context[0][11], 0.0);

        double[][] future = input.futureFeatures();
        assertEquals(0.0, future[0][0], 0.0);
        assertEquals(1.0, future[0][8], 0.0);
        assertFalse(input.futureCovariateMask()[1]);
    }

    @Test
    void truncatesContextToConfiguredMaximum() {
        var input = new Chronos2InputAssembler(config()).assemble(
                new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                null, null, null, 1);
        assertEquals(2, input.contextPatchCount());
        assertEquals(-1.0, input.contextFeatures()[0][0], 1e-12);
    }

    @Test
    void rejectsFutureMaskWithoutValuesAndHorizonOverflow() {
        var assembler = new Chronos2InputAssembler(config());
        assertThrows(IllegalArgumentException.class,
                () -> assembler.assemble(new double[] {1}, null, null, new boolean[1], 1));
        assertThrows(IllegalArgumentException.class,
                () -> assembler.assemble(new double[] {1}, null, new double[5], null, 1));
    }

    @Test
    void futureCovariatesUseContextStatistics() {
        var input = new Chronos2InputAssembler(config()).assemble(
                new double[] {1, 2, 3, 4}, null, new double[] {5}, null, 1);
        assertTrue(Double.isFinite(input.futureFeatures()[0][4]));
    }

    @Test
    void inputArraysAreDefensivelyCopied() {
        var input = new Chronos2InputAssembler(config()).assemble(
                new double[] {1, 2, 3, 4}, null, null, null, 1);
        double[][] changed = input.contextFeatures();
        changed[0][0] = 999;
        assertEquals(-0.5, input.contextFeatures()[0][0], 1e-12);
    }
}
