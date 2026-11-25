package tech.kayys.aljabr.models.common.transformer;

import java.util.Objects;

/**
 * Builds a same-group, per-time-step visibility mask for grouped time-series
 * transformer attention.
 *
 * <p>The returned layout is {@code [time][querySeries][keySeries]}. A key is
 * visible only when its series shares the query's group and has an observed
 * token at that time step. Query validity is deliberately not folded into the
 * mask; callers may use it to suppress padded query outputs separately.</p>
 */
public final class GroupAttentionMask {
    private GroupAttentionMask() {
    }

    public static boolean[][][] build(int[] groupIds, boolean[][] validTokensBySeries) {
        Objects.requireNonNull(groupIds, "groupIds");
        Objects.requireNonNull(validTokensBySeries, "validTokensBySeries");
        if (groupIds.length == 0 || validTokensBySeries.length != groupIds.length) {
            throw new IllegalArgumentException("group ids and validity rows must have the same non-zero series count");
        }
        if (validTokensBySeries[0] == null || validTokensBySeries[0].length == 0) {
            throw new IllegalArgumentException("validity rows must contain at least one time step");
        }
        int timeSteps = validTokensBySeries[0].length;
        for (boolean[] row : validTokensBySeries) {
            if (row == null || row.length != timeSteps) {
                throw new IllegalArgumentException("validity rows must have equal time dimensions");
            }
        }

        boolean[][][] visible = new boolean[timeSteps][groupIds.length][groupIds.length];
        for (int time = 0; time < timeSteps; time++) {
            for (int query = 0; query < groupIds.length; query++) {
                for (int key = 0; key < groupIds.length; key++) {
                    visible[time][query][key] =
                            groupIds[query] == groupIds[key] && validTokensBySeries[key][time];
                }
            }
        }
        return visible;
    }
}
