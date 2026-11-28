package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;

class VideoFrameSequenceTest {
    private static final Instant START = Instant.parse("2026-11-27T10:00:00Z");

    private static VideoFrameConditioningData frame(int index, long millis) {
        return new VideoFrameConditioningData("frame-" + index, index,
                START.plusMillis(millis),
                new DecodedVideoFrame(1, 1, 1, DecodedVideoFrame.Layout.HWC, new float[] {index}));
    }

    @Test
    void createsSequenceWithoutSortingAndAllowsSparseIndices() {
        var sequence = VideoFrameSequence.of(List.of(frame(0, 0), frame(3, 100), frame(8, 250)));

        assertEquals(3, sequence.frameCount());
        assertEquals(List.of(0, 3, 8),
                sequence.frames().stream().map(VideoFrameConditioningData::sequenceIndex).toList());
        assertEquals(Duration.ofMillis(250), sequence.duration());
        assertEquals(START, sequence.firstFrame().timestamp());
        assertEquals(START.plusMillis(250), sequence.lastFrame().timestamp());
        assertEquals(3, sequence.frameAtSequenceIndex(3).orElseThrow().sequenceIndex());
        assertTrue(sequence.frameAtSequenceIndex(4).isEmpty());
        assertThrows(UnsupportedOperationException.class, () -> sequence.frames().clear());
    }

    @Test
    void rejectsEmptySequence() {
        assertThrows(IllegalArgumentException.class, () -> VideoFrameSequence.of(List.of()));
    }

    @Test
    void rejectsOutOfOrderOrDuplicateSequenceIndices() {
        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSequence.of(List.of(frame(2, 0), frame(1, 100))));
        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSequence.of(List.of(frame(2, 0), frame(2, 100))));
    }

    @Test
    void rejectsEqualOrDecreasingTimestamps() {
        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSequence.of(List.of(frame(0, 0), frame(1, 0))));
        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSequence.of(List.of(frame(0, 100), frame(1, 50))));
    }

    @Test
    void extractsFramesFromPreparedSetWithoutTakingOwnership() throws Exception {
        var first = frame(0, 0);
        var second = frame(1, 100);
        var other = new StubData(DiffusionConditioningInput.Modality.PROMPT);
        var prepared = new PreparedDiffusionConditioningSet(List.of(first, other, second));

        var sequence = VideoFrameSequence.from(prepared);
        assertEquals(List.of(first, second), sequence.frames());

        // Sequence is a view: closing the prepared owner remains the one
        // authoritative lifecycle operation.
        prepared.close();
        assertTrue(other.closed);
    }

    private static final class StubData implements DiffusionConditioningData {
        private final DiffusionConditioningInput.Modality modality;
        private boolean closed;
        private StubData(DiffusionConditioningInput.Modality modality) { this.modality = modality; }
        @Override public String inputId() { return "prompt"; }
        @Override public DiffusionConditioningInput.Modality modality() { return modality; }
        @Override public void close() { closed = true; }
    }
}
