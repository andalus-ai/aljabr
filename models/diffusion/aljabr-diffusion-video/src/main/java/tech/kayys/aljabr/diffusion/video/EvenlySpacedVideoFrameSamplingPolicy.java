package tech.kayys.aljabr.diffusion.video;

import java.util.ArrayList;
import java.util.List;

/** Selects a fixed number of frames, including both endpoints when count exceeds one. */
public final class EvenlySpacedVideoFrameSamplingPolicy implements VideoFrameSamplingPolicy {
    private final int frameCount;

    public EvenlySpacedVideoFrameSamplingPolicy(int frameCount) {
        if (frameCount <= 0) {
            throw new IllegalArgumentException("frameCount must be positive");
        }
        this.frameCount = frameCount;
    }

    public int frameCount() {
        return frameCount;
    }

    @Override
    public List<VideoFrameConditioningData> select(VideoFrameSequence sequence) {
        int available = sequence.frameCount();
        if (frameCount > available) {
            throw new IllegalArgumentException(
                    "requested " + frameCount + " frames but only " + available + " are available");
        }

        List<VideoFrameConditioningData> selected = new ArrayList<>(frameCount);
        if (frameCount == 1) {
            // A single representative frame is the lower middle frame for deterministic ties.
            selected.add(sequence.frames().get((available - 1) / 2));
            return List.copyOf(selected);
        }

        long span = available - 1L;
        long denominator = frameCount - 1L;
        for (int i = 0; i < frameCount; i++) {
            int index = (int) ((i * span + denominator / 2) / denominator);
            selected.add(sequence.frames().get(index));
        }
        return List.copyOf(selected);
    }
}
