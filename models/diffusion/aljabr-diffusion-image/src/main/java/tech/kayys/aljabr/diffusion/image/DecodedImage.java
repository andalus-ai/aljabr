package tech.kayys.aljabr.diffusion.image;

import java.util.Objects;

/** Immutable backend-neutral decoded image pixels. Pixel range and color semantics are caller-defined. */
public final class DecodedImage {
    private final int width;
    private final int height;
    private final int channels;
    private final Layout layout;
    private final float[] pixels;

    public DecodedImage(int width, int height, int channels, Layout layout, float[] pixels) {
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw new IllegalArgumentException("image dimensions and channels must be positive");
        }
        this.layout = Objects.requireNonNull(layout, "layout");
        this.pixels = Objects.requireNonNull(pixels, "pixels").clone();
        long expected = (long) width * height * channels;
        if (expected != this.pixels.length) {
            throw new IllegalArgumentException("pixel count does not match image dimensions");
        }
        for (float pixel : this.pixels) {
            if (!Float.isFinite(pixel)) throw new IllegalArgumentException("pixel values must be finite");
        }
        this.width = width;
        this.height = height;
        this.channels = channels;
    }
    public int width() { return width; }
    public int height() { return height; }
    public int channels() { return channels; }
    public Layout layout() { return layout; }
    public float[] pixels() { return pixels.clone(); }
    public enum Layout { HWC, CHW }
}
