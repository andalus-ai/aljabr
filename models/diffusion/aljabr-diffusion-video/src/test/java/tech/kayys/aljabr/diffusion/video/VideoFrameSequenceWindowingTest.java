package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Instant;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class VideoFrameSequenceWindowingTest {
    private static final Instant START = Instant.parse("2026-11-27T10:00:00Z");

    private static VideoFrameSequence sequence(int count) {
        List<VideoFrameConditioningData> frames = IntStream.range(0, count)
                .mapToObj(index -> new VideoFrameConditioningData(
                        "video", index * 2, START.plusMillis(index * 40L),
                        new DecodedVideoFrame(1, 1, 1, DecodedVideoFrame.Layout.HWC,
                                new float[] {index})))
                .toList();
        return VideoFrameSequence.of(frames);
    }

    @Test
    void createsOverlappingWindowsWithoutChangingFrameProvenance() {
        var source = sequence(7);

        var windows = VideoFrameSequenceWindowing.window(
                source, 3, 2, VideoFrameSequenceWindowing.TailPolicy.DROP_PARTIAL);

        assertEquals(3, windows.size());
        assertEquals(List.of(0, 2, 4), windows.stream()
                .map(window -> window.firstFrame().sequenceIndex()).toList());
        assertEquals(List.of(4, 4, 4), windows.stream()
                .map(VideoFrameSequence::frameCount).toList());
        assertEquals(START.plusMillis(240), windows.get(2).lastFrame().timestamp());
    }

    @Test
    void createsNonOverlappingWindowsAndRetainsPartialTail() {
        var windows = VideoFrameSequenceWindowing.window(
                sequence(5), 2, 2, VideoFrameSequenceWindowing.TailPolicy.INCLUDE_PARTIAL);

        assertEquals(List.of(2, 2, 1), windows.stream()
                .map(VideoFrameSequence::frameCount).toList());
        assertEquals(8, windows.get(2).firstFrame().sequenceIndex());
    }

    @Test
    void canSkipFramesBetweenWindows() {
        var windows = VideoFrameSequenceWindowing.window(
                sequence(8), 2, 3, VideoFrameSequenceWindowing.TailPolicy.DROP_PARTIAL);

        assertEquals(List.of(0, 6), windows.stream()
                .map(window -> window.firstFrame().sequenceIndex()).toList());
        assertEquals(2, windows.size());
    }

    @Test
    void dropsIncompleteTailWhenRequested() {
        var windows = VideoFrameSequenceWindowing.window(
                sequence(5), 3, 3, VideoFrameSequenceWindowing.TailPolicy.DROP_PARTIAL);

        assertEquals(1, windows.size());
        assertEquals(3, windows.getFirst().frameCount());
    }

    @Test
    void returnsNoWindowsWhenNoCompleteWindowCanBeFormed() {
        var windows = VideoFrameSequenceWindowing.window(
                sequence(2), 3, 1, VideoFrameSequenceWindowing.TailPolicy.DROP_PARTIAL);

        assertTrue(windows.isEmpty());
    }

    @Test
    void rejectsNonPositiveWindowSizeAndStride() {
        var source = sequence(3);

        assertThrows(IllegalArgumentException.class, () -> VideoFrameSequenceWindowing.window(
                source, 0, 1, VideoFrameSequenceWindowing.TailPolicy.INCLUDE_PARTIAL));
        assertThrows(IllegalArgumentException.class, () -> VideoFrameSequenceWindowing.window(
                source, 1, 0, VideoFrameSequenceWindowing.TailPolicy.INCLUDE_PARTIAL));
    }

    @Test
    void windowsAndTheirFrameListsAreImmutable() {
        var windows = VideoFrameSequenceWindowing.window(
                sequence(3), 2, 1, VideoFrameSequenceWindowing.TailPolicy.INCLUDE_PARTIAL);

        assertThrows(UnsupportedOperationException.class, () -> windows.clear());
        assertThrows(UnsupportedOperationException.class, () -> windows.getFirst().frames().clear());
    }
}
