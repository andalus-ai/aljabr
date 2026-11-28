package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

class VideoFrameSamplingPolicyTest {
    private static final Instant START = Instant.parse("2026-11-27T10:00:00Z");

    private static VideoFrameConditioningData frame(int index, long millis) {
        return new VideoFrameConditioningData("frame-" + index, index,
                START.plusMillis(millis),
                new DecodedVideoFrame(1, 1, 1, DecodedVideoFrame.Layout.HWC, new float[] {index}));
    }

    private static VideoFrameSequence sequence(long... timestamps) {
        var frames = new java.util.ArrayList<VideoFrameConditioningData>();
        for (int i = 0; i < timestamps.length; i++) {
            frames.add(frame(i, timestamps[i]));
        }
        return VideoFrameSequence.of(frames);
    }

    @Test
    void evenlySpacedSamplingIncludesEndpointsAndPreservesOriginalFrames() {
        var source = sequence(0, 40, 100, 170, 250);

        var sampled = VideoFrameSampler.sample(source,
                new EvenlySpacedVideoFrameSamplingPolicy(3)).orElseThrow();

        assertEquals(List.of(0, 2, 4),
                sampled.frames().stream().map(VideoFrameConditioningData::sequenceIndex).toList());
        assertSame(source.firstFrame(), sampled.firstFrame());
        assertSame(source.lastFrame(), sampled.lastFrame());
    }

    @Test
    void oneEvenlySpacedFrameSelectsTheLowerMiddleFrame() {
        var sampled = VideoFrameSampler.sample(sequence(0, 10, 20, 30),
                new EvenlySpacedVideoFrameSamplingPolicy(1)).orElseThrow();

        assertEquals(1, sampled.firstFrame().sequenceIndex());
    }

    @Test
    void evenlySpacedSamplingRejectsInvalidCounts() {
        assertThrows(IllegalArgumentException.class, () -> new EvenlySpacedVideoFrameSamplingPolicy(0));
        assertThrows(IllegalArgumentException.class, () -> VideoFrameSampler.sample(
                sequence(0, 10), new EvenlySpacedVideoFrameSamplingPolicy(3)));
    }

    @Test
    void intervalSamplingUsesFixedTargetsAndSkipsMissedTargets() {
        var sampled = VideoFrameSampler.sample(sequence(0, 40, 110, 250, 310),
                new TimestampIntervalVideoFrameSamplingPolicy(Duration.ofMillis(100))).orElseThrow();

        // Targets are 0, 100, 200, 300 ms; select the first frame at/after each.
        assertEquals(List.of(0, 2, 3, 4),
                sampled.frames().stream().map(VideoFrameConditioningData::sequenceIndex).toList());
    }

    @Test
    void intervalSamplingRequiresPositiveDuration() {
        assertThrows(IllegalArgumentException.class,
                () -> new TimestampIntervalVideoFrameSamplingPolicy(Duration.ZERO));
        assertThrows(IllegalArgumentException.class,
                () -> new TimestampIntervalVideoFrameSamplingPolicy(Duration.ofMillis(-1)));
    }

    @Test
    void timestampWindowUsesInclusiveBoundariesAndMaySelectNoFrames() {
        var source = sequence(0, 100, 200, 300);
        var policy = new TimestampWindowVideoFrameSamplingPolicy(
                START.plusMillis(100), START.plusMillis(200));

        var sampled = VideoFrameSampler.sample(source, policy).orElseThrow();
        assertEquals(List.of(1, 2),
                sampled.frames().stream().map(VideoFrameConditioningData::sequenceIndex).toList());

        assertTrue(VideoFrameSampler.sample(source,
                new TimestampWindowVideoFrameSamplingPolicy(
                        START.plusMillis(400), START.plusMillis(500))).isEmpty());
    }

    @Test
    void timestampWindowRejectsReversedBounds() {
        assertThrows(IllegalArgumentException.class,
                () -> new TimestampWindowVideoFrameSamplingPolicy(
                        START.plusSeconds(2), START.plusSeconds(1)));
    }

    @Test
    void samplerRejectsForeignOrReorderedFramesFromCustomPolicies() {
        var source = sequence(0, 100, 200);
        var foreign = frame(9, 900);

        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSampler.sample(source, ignored -> List.of(source.firstFrame(), foreign)));
        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSampler.sample(source, ignored -> List.of(source.lastFrame(), source.firstFrame())));
        assertThrows(IllegalArgumentException.class,
                () -> VideoFrameSampler.sample(source, ignored -> List.of(source.firstFrame(), source.firstFrame())));
    }

    @Test
    void samplerReturnsEmptyForAnEmptyPolicySelection() {
        assertTrue(VideoFrameSampler.sample(sequence(0, 100), ignored -> List.of()).isEmpty());
    }
}
