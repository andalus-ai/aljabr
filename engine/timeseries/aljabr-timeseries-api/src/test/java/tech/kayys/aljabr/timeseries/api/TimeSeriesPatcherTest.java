package tech.kayys.aljabr.timeseries.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class TimeSeriesPatcherTest {
    @Test
    void leftPadsToPatchBoundaryAndPreservesMissingMask() {
        List<TimeSeriesPatcher.Patch> patches =
                new TimeSeriesPatcher(4, 4).patch(new double[] {1, 2, 3, 4, 5});
        assertEquals(2, patches.size());
        assertEquals(-3, patches.getFirst().startIndex());
        assertTrue(Double.isNaN(patches.getFirst().values()[0]));
        assertFalse(patches.getFirst().observedMask()[0]);
        assertEquals(1.0, patches.getFirst().values()[3]);
        assertEquals(5.0, patches.getLast().values()[3]);
    }

    @Test
    void supportsOverlappingWindows() {
        List<TimeSeriesPatcher.Patch> patches =
                new TimeSeriesPatcher(4, 2).patch(new double[] {1, 2, 3, 4, 5, 6, 7, 8});
        assertEquals(3, patches.size());
        assertEquals(2, patches.get(1).startIndex());
        assertEquals(3.0, patches.get(1).values()[0]);
    }

    @Test
    void retainsAllMissingPatchAsMasked() {
        var patches = new TimeSeriesPatcher(2, 2).patch(new double[] {Double.NaN, Double.NaN});
        assertFalse(patches.getFirst().hasObservedValues());
    }
}
