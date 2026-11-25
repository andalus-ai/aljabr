package tech.kayys.aljabr.models.common.transformer;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GroupAttentionMaskTest {
    @Test
    void restrictsKeysToSameGroupAndValidTimePositions() {
        int[] groups = {7, 7, 9};
        boolean[][] valid = {
            {true, false},
            {true, true},
            {true, true}
        };

        boolean[][][] mask = GroupAttentionMask.build(groups, valid);
        assertTrue(mask[0][0][0]);
        assertTrue(mask[0][0][1]);
        assertFalse(mask[0][0][2]);
        assertFalse(mask[1][0][1]);
        assertTrue(mask[1][1][1]);
        assertTrue(mask[1][2][2]);
    }

    @Test
    void rejectsMismatchedOrRaggedInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> GroupAttentionMask.build(new int[] {1, 2}, new boolean[][] {{true}}));
        assertThrows(IllegalArgumentException.class,
                () -> GroupAttentionMask.build(new int[] {1, 2}, new boolean[][] {{true}, {true, false}}));
    }
}
