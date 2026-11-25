package tech.kayys.aljabr.diffusion.minimaxh3;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningSet;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MiniMaxH3ConditioningCapabilitiesTest {

    @Test
    void acceptsOrderedReferenceFramesForMultiframeTask() {
        var inputs = new DiffusionConditioningSet(List.of(
                new DiffusionConditioningInput("frame-0", "file:///0.png", 0),
                new DiffusionConditioningInput("frame-1", "file:///1.png", 1),
                new DiffusionConditioningInput("frame-2", "file:///2.png", 2)));

        assertDoesNotThrow(() ->
                MiniMaxH3ConditioningCapabilities.validate(
                        MiniMaxH3Task.MULTIFRAME_REFERENCE_TO_VIDEO, inputs));
    }

    @Test
    void rejectsVideoFramesForSingleImageToVideoTask() {
        var inputs = new DiffusionConditioningSet(List.of(
                new DiffusionConditioningInput("frame-0", "file:///0.png", 0)));

        assertThrows(IllegalArgumentException.class, () ->
                MiniMaxH3ConditioningCapabilities.validate(MiniMaxH3Task.IMAGE_TO_VIDEO, inputs));
    }

    @Test
    void acceptsSingleImageForImageToVideoAndEdit() {
        var inputs = new DiffusionConditioningSet(List.of(
                new DiffusionConditioningInput(
                        "source", DiffusionConditioningInput.Modality.IMAGE, "file:///source.png")));

        assertDoesNotThrow(() ->
                MiniMaxH3ConditioningCapabilities.validate(MiniMaxH3Task.IMAGE_TO_VIDEO, inputs));
        assertDoesNotThrow(() ->
                MiniMaxH3ConditioningCapabilities.validate(MiniMaxH3Task.IMAGE_EDIT, inputs));
    }

    @Test
    void rejectsImageInputForTextOnlyTask() {
        var inputs = new DiffusionConditioningSet(List.of(
                new DiffusionConditioningInput(
                        "source", DiffusionConditioningInput.Modality.IMAGE, "file:///source.png")));

        assertThrows(IllegalArgumentException.class, () ->
                MiniMaxH3ConditioningCapabilities.validate(MiniMaxH3Task.TEXT_TO_VIDEO, inputs));
    }

    @Test
    void permitsEmptyConditioningSetWithoutImplyingInputRequirements() {
        assertDoesNotThrow(() ->
                MiniMaxH3ConditioningCapabilities.validate(
                        MiniMaxH3Task.TEXT_TO_VIDEO, new DiffusionConditioningSet(List.of())));
    }
}
