package tech.kayys.aljabr.diffusion.video;
import java.util.Objects;
/** Immutable backend-neutral pixels for one decoded video frame. */
public final class DecodedVideoFrame {
    private final int width, height, channels;
    private final Layout layout;
    private final float[] pixels;
    public DecodedVideoFrame(int width, int height, int channels, Layout layout, float[] pixels) {
        if (width <= 0 || height <= 0 || channels <= 0) throw new IllegalArgumentException("frame dimensions and channels must be positive");
        this.layout = Objects.requireNonNull(layout, "layout");
        this.pixels = Objects.requireNonNull(pixels, "pixels").clone();
        if ((long) width * height * channels != this.pixels.length) throw new IllegalArgumentException("pixel count does not match frame dimensions");
        for (float pixel : this.pixels) if (!Float.isFinite(pixel)) throw new IllegalArgumentException("pixel values must be finite");
        this.width = width; this.height = height; this.channels = channels;
    }
    public int width() { return width; }
    public int height() { return height; }
    public int channels() { return channels; }
    public Layout layout() { return layout; }
    public float[] pixels() { return pixels.clone(); }
    public enum Layout { HWC, CHW }
}
