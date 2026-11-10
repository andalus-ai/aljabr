package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.minFromPackedScaleMin;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.scaleFromPackedScaleMin;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.scaleK4Packed;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.scaleMinK4PackedCode;

/**
 * Prepared K-quant scale/min expansion for Q4_K and Q5_K matrices.
 * @author bhangun
 */
final class GgufKScales {
    private GgufKScales() {
    }

    static void fill(
            long scalesLow,
            int scalesHigh,
            float d,
            float dMin,
            boolean mayHaveGroupMins,
            float[] groupScales,
            State state,
            int groupBase) {
        if (mayHaveGroupMins && dMin != 0.0f) {
            fillWithMins(scalesLow, scalesHigh, d, dMin, groupScales, state, groupBase);
        } else {
            fillPlain(scalesLow, scalesHigh, d, groupScales, groupBase);
        }
    }

    private static void fillPlain(
            long scalesLow,
            int scalesHigh,
            float d,
            float[] groupScales,
            int groupBase) {
        int scaleIndex = 0;
        int groupIndex = groupBase;
        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            fillPairPlain(scalesLow, scalesHigh, scaleIndex, d, groupScales, groupIndex);
            scaleIndex += 2;
            groupIndex += 2;
        }
    }

    private static void fillWithMins(
            long scalesLow,
            int scalesHigh,
            float d,
            float dMin,
            float[] groupScales,
            State state,
            int groupBase) {
        int scaleIndex = 0;
        int groupIndex = groupBase;
        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            fillPairWithMins(scalesLow, scalesHigh, scaleIndex, d, dMin, groupScales, state, groupIndex);
            scaleIndex += 2;
            groupIndex += 2;
        }
    }

    private static void fillPairPlain(
            long scalesLow,
            int scalesHigh,
            int scaleIndex,
            float d,
            float[] groupScales,
            int groupIndex) {
        groupScales[groupIndex] = d * scaleK4Packed(scalesLow, scalesHigh, scaleIndex);
        groupScales[groupIndex + 1] = d * scaleK4Packed(scalesLow, scalesHigh, scaleIndex + 1);
    }

    private static void fillPairWithMins(
            long scalesLow,
            int scalesHigh,
            int scaleIndex,
            float d,
            float dMin,
            float[] groupScales,
            State state,
            int groupIndex) {
        int first = scaleMinK4PackedCode(scalesLow, scalesHigh, scaleIndex);
        int second = scaleMinK4PackedCode(scalesLow, scalesHigh, scaleIndex + 1);
        int firstMin = minFromPackedScaleMin(first);
        int secondMin = minFromPackedScaleMin(second);
        groupScales[groupIndex] = d * scaleFromPackedScaleMin(first);
        groupScales[groupIndex + 1] = d * scaleFromPackedScaleMin(second);
        if (firstMin == 0 && secondMin == 0) {
            return;
        }
        if (state.groupMins == null) {
            state.groupMins = new float[groupScales.length];
        }
        state.groupMins[groupIndex] = dMin * firstMin;
        state.groupMins[groupIndex + 1] = dMin * secondMin;
        state.hasGroupMins = true;
    }

    static final class State {
        float[] groupMins;
        boolean hasGroupMins;
    }
}
