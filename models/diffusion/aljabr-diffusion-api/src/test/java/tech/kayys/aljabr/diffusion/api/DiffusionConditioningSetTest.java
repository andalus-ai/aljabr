package tech.kayys.aljabr.diffusion.api;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionConditioningSetTest {

    @Test
    void preservesInputOrderAndFiltersByModality() {
        var frame1 = new DiffusionConditioningInput("frame-1", "file:///frame-1.png", 0);
        var frame2 = new DiffusionConditioningInput("frame-2", "file:///frame-2.png", 1);
        var prompt = new DiffusionConditioningInput(
                "prompt", DiffusionConditioningInput.Modality.PROMPT, "text:walk forward");

        var inputs = new DiffusionConditioningSet(List.of(frame1, frame2, prompt));

        assertEquals(List.of(frame1, frame2, prompt), inputs.inputs());
        assertEquals(List.of(frame1, frame2),
                inputs.inputsOf(DiffusionConditioningInput.Modality.VIDEO_FRAME));
    }

    @Test
    void rejectsDuplicateInputIds() {
        var first = new DiffusionConditioningInput(
                "same", DiffusionConditioningInput.Modality.IMAGE, "file:///one.png");
        var second = new DiffusionConditioningInput(
                "same", DiffusionConditioningInput.Modality.IMAGE, "file:///two.png");

        assertThrows(IllegalArgumentException.class,
                () -> new DiffusionConditioningSet(List.of(first, second)));
    }

    @Test
    void rejectsDuplicateFrameIndices() {
        var first = new DiffusionConditioningInput("frame-1", "file:///one.png", 3);
        var second = new DiffusionConditioningInput("frame-2", "file:///two.png", 3);

        assertThrows(IllegalArgumentException.class,
                () -> new DiffusionConditioningSet(List.of(first, second)));
    }

    @Test
    void requiresSequenceIndexOnlyForVideoFrames() {
        assertThrows(IllegalArgumentException.class, () ->
                new DiffusionConditioningInput(
                        "still", DiffusionConditioningInput.Modality.IMAGE,
                        "file:///still.png", 0, java.util.Map.of()));
        assertThrows(IllegalArgumentException.class, () ->
                new DiffusionConditioningInput(
                        "frame", DiffusionConditioningInput.Modality.VIDEO_FRAME,
                        "file:///frame.png"));
    }
}
