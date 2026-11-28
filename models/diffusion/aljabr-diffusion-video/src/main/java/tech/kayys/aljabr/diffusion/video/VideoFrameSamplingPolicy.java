package tech.kayys.aljabr.diffusion.video;

import java.util.List;

/** Strategy for selecting an ordered subset of frames from a prepared sequence. */
@FunctionalInterface
public interface VideoFrameSamplingPolicy {
    /**
     * Selects frames without decoding, copying, interpolating, or taking ownership.
     * The result must be an ordered subset of the supplied sequence.
     */
    List<VideoFrameConditioningData> select(VideoFrameSequence sequence);
}
