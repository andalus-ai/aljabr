package tech.kayys.aljabr.diffusion.video;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Packs a validated frame sequence into contiguous backend-neutral heap memory. */
public final class VideoFrameBatchPacker {
    private VideoFrameBatchPacker() {
    }

    /**
     * Packs all frames into the requested layout, converting per-frame HWC/CHW
     * storage as necessary. All frames must have identical width, height, and
     * channel count.
     *
     * <p>The returned batch owns a new copy and is independent of the source
     * sequence and its conditioning-set lifecycle.</p>
     */
    public static VideoFrameBatch pack(VideoFrameSequence sequence, VideoFrameBatchLayout layout) {
        Objects.requireNonNull(sequence, "sequence");
        Objects.requireNonNull(layout, "layout");

        VideoFrameConditioningData first = sequence.firstFrame();
        int width = first.width();
        int height = first.height();
        int channels = first.channels();
        long valueCount;
        try {
            long valuesPerFrame = Math.multiplyExact(Math.multiplyExact((long) height, width), channels);
            valueCount = Math.multiplyExact((long) sequence.frameCount(), valuesPerFrame);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("packed video batch dimensions overflow", overflow);
        }
        if (valueCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("packed video batch exceeds Java array capacity");
        }

        float[] packed = new float[(int) valueCount];
        List<VideoFrameBatchMetadata> metadata = new ArrayList<>(sequence.frameCount());
        for (int t = 0; t < sequence.frameCount(); t++) {
            VideoFrameConditioningData frame = sequence.frames().get(t);
            if (frame.width() != width || frame.height() != height || frame.channels() != channels) {
                throw new IllegalArgumentException(
                        "all frames must have identical width, height, and channel count; frame "
                                + frame.sequenceIndex() + " differs");
            }
            float[] source = frame.pixels();
            boolean sourceHwc = frame.layout() == DecodedVideoFrame.Layout.HWC;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        int sourceIndex = sourceHwc
                                ? ((y * width + x) * channels + c)
                                : ((c * height + y) * width + x);
                        int targetIndex = layout == VideoFrameBatchLayout.THWC
                                ? (((t * height + y) * width + x) * channels + c)
                                : (((t * channels + c) * height + y) * width + x);
                        packed[targetIndex] = source[sourceIndex];
                    }
                }
            }
            metadata.add(VideoFrameBatchMetadata.from(frame));
        }
        return new VideoFrameBatch(sequence.frameCount(), height, width, channels,
                layout, packed, metadata);
    }
}
