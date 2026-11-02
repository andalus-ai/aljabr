/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizedTensorInfo.java
 * ───────────────────────
 * Quantized tensor metadata.
 */
package tech.kayys.aljabr.safetensor.quantization;

/**
 * Metadata for a quantized tensor.
 *
 * <p>Array-valued fields use defensive copies so callers cannot mutate metadata
 * after construction or through an accessor.</p>
 *
 * @author bhangun
 * @version 1.0.0
 */
public class QuantizedTensorInfo {
    private final String name;
    private final int[] shape;
    private final String originalDtype;
    private final String quantizedDtype;
    private final QuantizationEngine.QuantStrategy strategy;
    private final float[] scales;
    private final float[] zeros;
    private final int groupSize;
    private final int channelAxis;
    private final byte[] metadata;

    private QuantizedTensorInfo(Builder builder) {
        this.name = builder.name;
        this.shape = copy(builder.shape);
        this.originalDtype = builder.originalDtype;
        this.quantizedDtype = builder.quantizedDtype;
        this.strategy = builder.strategy;
        this.scales = copy(builder.scales);
        this.zeros = copy(builder.zeros);
        this.groupSize = builder.groupSize;
        this.channelAxis = builder.channelAxis;
        this.metadata = copy(builder.metadata);
    }

    private static int[] copy(int[] value) {
        return value == null ? null : value.clone();
    }

    private static float[] copy(float[] value) {
        return value == null ? null : value.clone();
    }

    private static byte[] copy(byte[] value) {
        return value == null ? null : value.clone();
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getName() { return name; }
    public int[] getShape() { return copy(shape); }
    public String getOriginalDtype() { return originalDtype; }
    public String getQuantizedDtype() { return quantizedDtype; }
    public QuantizationEngine.QuantStrategy getStrategy() { return strategy; }
    public float[] getScales() { return copy(scales); }
    public float[] getZeros() { return copy(zeros); }
    public int getGroupSize() { return groupSize; }
    public int getChannelAxis() { return channelAxis; }
    public byte[] getMetadata() { return copy(metadata); }

    /** Builder for QuantizedTensorInfo. */
    public static class Builder {
        private String name;
        private int[] shape;
        private String originalDtype = "F32";
        private String quantizedDtype = "INT8";
        private QuantizationEngine.QuantStrategy strategy = QuantizationEngine.QuantStrategy.INT8;
        private float[] scales = new float[0];
        private float[] zeros = new float[0];
        private int groupSize = 128;
        private int channelAxis = 0;
        private byte[] metadata = new byte[0];

        public Builder name(String name) { this.name = name; return this; }
        public Builder shape(int[] shape) { this.shape = copy(shape); return this; }
        public Builder originalDtype(String originalDtype) { this.originalDtype = originalDtype; return this; }
        public Builder quantizedDtype(String quantizedDtype) { this.quantizedDtype = quantizedDtype; return this; }
        public Builder strategy(QuantizationEngine.QuantStrategy strategy) { this.strategy = strategy; return this; }
        public Builder scales(float[] scales) { this.scales = copy(scales); return this; }
        public Builder zeros(float[] zeros) { this.zeros = copy(zeros); return this; }
        public Builder groupSize(int groupSize) { this.groupSize = groupSize; return this; }
        public Builder channelAxis(int channelAxis) { this.channelAxis = channelAxis; return this; }
        public Builder metadata(byte[] metadata) { this.metadata = copy(metadata); return this; }
        public QuantizedTensorInfo build() { return new QuantizedTensorInfo(this); }
    }
}
