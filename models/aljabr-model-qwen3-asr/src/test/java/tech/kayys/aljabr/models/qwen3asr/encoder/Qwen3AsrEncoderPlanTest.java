package tech.kayys.aljabr.models.qwen3asr.encoder;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.qwen3asr.Qwen3AsrConfig;

import static org.junit.jupiter.api.Assertions.*;

class Qwen3AsrEncoderPlanTest {
    private final Qwen3AsrEncoderPlan planner =
            new Qwen3AsrEncoderPlan(Qwen3AsrConfig.qwen3Asr17B().audio());

    @Test
    void convolutionLengthMatchesThreeStrideTwoPaddedStages() {
        assertEquals(13, Qwen3AsrEncoderPlan.convolutionOutputLength(100));
        assertEquals(7, Qwen3AsrEncoderPlan.convolutionOutputLength(50));
        assertEquals(1, Qwen3AsrEncoderPlan.convolutionOutputLength(1));
        assertEquals(375, Qwen3AsrEncoderPlan.convolutionOutputLength(3_000));
    }

    @Test
    void oneSecondOfMelFramesFormsOneConvolutionChunkAndOneAttentionWindow() {
        var plan = planner.plan(100);

        assertEquals(1, plan.convolutionChunks().size());
        assertEquals(1, plan.attentionWindows().size());
        assertEquals(13, plan.totalOutputTokens());
        assertEquals(0, plan.convolutionChunks().getFirst().positionalOffset());
        assertEquals(0, plan.convolutionChunks().getFirst().attentionWindowIndex());
        assertEquals(100, plan.convolutionChunks().getFirst().inputFrameCount());
    }

    @Test
    void eightConvolutionChunksShareOneInferenceAttentionWindow() {
        var plan = planner.plan(800);

        assertEquals(8, plan.convolutionChunks().size());
        assertEquals(1, plan.attentionWindows().size());
        assertEquals(104, plan.totalOutputTokens());
        assertEquals(8, plan.attentionWindows().getFirst().chunkCount());
        assertEquals(104, plan.attentionWindows().getFirst().tokenCount());
    }

    @Test
    void longInputIsSplitIntoIndependentAttentionWindows() {
        var plan = planner.plan(3_000);

        assertEquals(30, plan.convolutionChunks().size());
        assertEquals(4, plan.attentionWindows().size());
        assertEquals(390, plan.totalOutputTokens());
        assertEquals(8, plan.attentionWindows().get(0).chunkCount());
        assertEquals(8, plan.attentionWindows().get(1).chunkCount());
        assertEquals(8, plan.attentionWindows().get(2).chunkCount());
        assertEquals(6, plan.attentionWindows().get(3).chunkCount());
        assertEquals(3, plan.attentionWindows().get(3).index());
        assertEquals(3, plan.convolutionChunks().get(24).attentionWindowIndex());
    }

    @Test
    void finalPartialConvolutionChunkUsesItsActualInputLength() {
        var plan = planner.plan(225);

        assertEquals(3, plan.convolutionChunks().size());
        assertEquals(100, plan.convolutionChunks().get(0).inputFrameCount());
        assertEquals(100, plan.convolutionChunks().get(1).inputFrameCount());
        assertEquals(25, plan.convolutionChunks().get(2).inputFrameCount());
        assertEquals(4, plan.convolutionChunks().get(2).outputTokenCount());
        assertEquals(30, plan.totalOutputTokens());
    }

    @Test
    void rejectsEmptyOrNegativeFeatureLengths() {
        assertThrows(IllegalArgumentException.class, () -> planner.plan(0));
        assertThrows(IllegalArgumentException.class, () -> planner.plan(-1));
        assertThrows(IllegalArgumentException.class,
                () -> Qwen3AsrEncoderPlan.convolutionOutputLength(0));
    }

    @Test
    void planCollectionsAreImmutable() {
        var plan = planner.plan(100);
        assertThrows(UnsupportedOperationException.class,
                () -> plan.convolutionChunks().clear());
        assertThrows(UnsupportedOperationException.class,
                () -> plan.attentionWindows().clear());
    }
}
