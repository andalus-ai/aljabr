package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

class VideoFrameBatchPackerTest {
    private static final Instant START = Instant.parse("2026-11-27T10:00:00Z");

    private static VideoFrameConditioningData frame(
            String id, int index, long millis, int width, int height, int channels,
            DecodedVideoFrame.Layout layout, float... values) {
        return new VideoFrameConditioningData(id, index, START.plusMillis(millis),
                new DecodedVideoFrame(width, height, channels, layout, values));
    }

    @Test
    void packsHwcFramesIntoThwcAndRetainsTemporalProvenance() {
        var first = frame("video-a", 2, 0, 2, 1, 2,
                DecodedVideoFrame.Layout.HWC, 1, 2, 3, 4);
        var second = frame("video-a", 5, 100, 2, 1, 2,
                DecodedVideoFrame.Layout.HWC, 5, 6, 7, 8);

        var batch = VideoFrameBatchPacker.pack(
                VideoFrameSequence.of(List.of(first, second)), VideoFrameBatchLayout.THWC);

        assertEquals(2, batch.frameCount());
        assertEquals(2, batch.width());
        assertEquals(1, batch.height());
        assertEquals(2, batch.channels());
        assertEquals(VideoFrameBatchLayout.THWC, batch.layout());
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, batch.values());
        assertEquals(7, batch.valueAt(1, 0, 1, 1));
        assertEquals(List.of(
                new VideoFrameBatchMetadata("video-a", 2, START),
                new VideoFrameBatchMetadata("video-a", 5, START.plusMillis(100))),
                batch.metadata());
    }

    @Test
    void convertsChwSourceFramesIntoThwcOutput() {
        var source = frame("video", 0, 0, 2, 1, 2,
                DecodedVideoFrame.Layout.CHW, 1, 2, 3, 4);

        var batch = VideoFrameBatchPacker.pack(
                VideoFrameSequence.of(List.of(source)), VideoFrameBatchLayout.THWC);

        assertArrayEquals(new float[] {1, 3, 2, 4}, batch.values());
        assertEquals(3, batch.valueAt(0, 0, 0, 1));
        assertEquals(4, batch.valueAt(0, 0, 1, 1));
    }

    @Test
    void convertsHwcSourceFramesIntoTchwOutput() {
        var source = frame("video", 0, 0, 2, 1, 2,
                DecodedVideoFrame.Layout.HWC, 1, 3, 2, 4);

        var batch = VideoFrameBatchPacker.pack(
                VideoFrameSequence.of(List.of(source)), VideoFrameBatchLayout.TCHW);

        assertArrayEquals(new float[] {1, 2, 3, 4}, batch.values());
        assertEquals(2, batch.valueAt(0, 0, 1, 0));
        assertEquals(4, batch.valueAt(0, 0, 1, 1));
    }

    @Test
    void batchValuesAreDefensiveAndIndependentFromSourceFrames() {
        var source = frame("video", 0, 0, 1, 1, 1,
                DecodedVideoFrame.Layout.HWC, 9);
        var batch = VideoFrameBatchPacker.pack(
                VideoFrameSequence.of(List.of(source)), VideoFrameBatchLayout.THWC);

        float[] exposed = batch.values();
        exposed[0] = -100;

        assertEquals(9, batch.valueAt(0, 0, 0, 0));
        assertThrows(UnsupportedOperationException.class, () -> batch.metadata().clear());
    }

    @Test
    void rejectsFramesWithDifferentGeometryOrChannelCount() {
        var first = frame("video", 0, 0, 2, 1, 1,
                DecodedVideoFrame.Layout.HWC, 1, 2);
        var differentWidth = frame("video", 1, 100, 1, 1, 1,
                DecodedVideoFrame.Layout.HWC, 3);

        assertThrows(IllegalArgumentException.class, () -> VideoFrameBatchPacker.pack(
                VideoFrameSequence.of(List.of(first, differentWidth)), VideoFrameBatchLayout.THWC));
    }

    @Test
    void rejectsOutOfBoundsLogicalCoordinates() {
        var batch = VideoFrameBatchPacker.pack(
                VideoFrameSequence.of(List.of(frame("video", 0, 0, 1, 1, 1,
                        DecodedVideoFrame.Layout.HWC, 1))), VideoFrameBatchLayout.THWC);

        assertThrows(IndexOutOfBoundsException.class, () -> batch.valueAt(1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> batch.valueAt(0, 1, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> batch.valueAt(0, 0, 0, 1));
    }
}
