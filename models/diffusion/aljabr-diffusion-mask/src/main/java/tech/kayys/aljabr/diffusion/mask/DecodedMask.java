package tech.kayys.aljabr.diffusion.mask;

import java.util.Objects;

/** Immutable, single-channel, row-major mask values in the inclusive range [0, 1]. */
public final class DecodedMask {
    private final int width;
    private final int height;
    private final float[] values;

    public DecodedMask(int width, int height, float[] values) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("mask dimensions must be positive");
        this.values = Objects.requireNonNull(values, "values").clone();
        if ((long) width * height != this.values.length) {
            throw new IllegalArgumentException("mask value count does not match dimensions");
        }
        for (float value : this.values) {
            if (!Float.isFinite(value) || value < 0f || value > 1f) {
                throw new IllegalArgumentException("mask values must be finite and in [0, 1]");
            }
        }
        this.width = width;
        this.height = height;
    }
    public int width() { return width; }
    public int height() { return height; }
    public float[] values() { return values.clone(); }
}
