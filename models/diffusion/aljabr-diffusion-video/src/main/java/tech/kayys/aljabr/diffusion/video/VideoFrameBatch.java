package tech.kayys.aljabr.diffusion.video;

import java.util.List;
import java.util.Objects;

/**
 * Immutable, contiguous, backend-neutral packed frames.
 *
 * <p>The batch owns an independent heap copy of the packed values. It remains
 * valid after the source conditioning set is closed. Values returned to callers
 * are defensive copies; use {@link #valueAt(int, int, int, int)} for scalar
 * access without exposing the backing array.</p>
 */
public final class VideoFrameBatch {
    private final int frameCount;
    private final int height;
    private final int width;
    private final int channels;
    private final VideoFrameBatchLayout layout;
    private final float[] values;
    private final List<VideoFrameBatchMetadata> metadata;

    VideoFrameBatch(int frameCount, int height, int width, int channels,
            VideoFrameBatchLayout layout, float[] values,
            List<VideoFrameBatchMetadata> metadata) {
        if (frameCount <= 0 || height <= 0 || width <= 0 || channels <= 0) {
            throw new IllegalArgumentException("batch dimensions must be positive");
        }
        this.layout = Objects.requireNonNull(layout, "layout");
        this.metadata = List.copyOf(Objects.requireNonNull(metadata, "metadata"));
        if (this.metadata.size() != frameCount) {
            throw new IllegalArgumentException("metadata count must match frame count");
        }
        long expected = (long) frameCount * height * width * channels;
        if (expected > Integer.MAX_VALUE || values.length != (int) expected) {
            throw new IllegalArgumentException("packed value count does not match batch dimensions");
        }
        this.frameCount = frameCount;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.values = Objects.requireNonNull(values, "values").clone();
        for (float value : this.values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("packed values must be finite");
            }
        }
    }

    public int frameCount() { return frameCount; }
    public int height() { return height; }
    public int width() { return width; }
    public int channels() { return channels; }
    public VideoFrameBatchLayout layout() { return layout; }
    public List<VideoFrameBatchMetadata> metadata() { return metadata; }

    /** Number of scalar values in the contiguous batch. */
    public int valueCount() { return values.length; }

    /** Returns a defensive copy of the contiguous packed values. */
    public float[] values() { return values.clone(); }

    /** Reads one scalar using logical [time, height, width, channel] coordinates. */
    public float valueAt(int time, int y, int x, int channel) {
        if (time < 0 || time >= frameCount || y < 0 || y >= height
                || x < 0 || x >= width || channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("video batch coordinate out of bounds");
        }
        int index = layout == VideoFrameBatchLayout.THWC
                ? (((time * height + y) * width + x) * channels + channel)
                : (((time * channels + channel) * height + y) * width + x);
        return values[index];
    }
}
