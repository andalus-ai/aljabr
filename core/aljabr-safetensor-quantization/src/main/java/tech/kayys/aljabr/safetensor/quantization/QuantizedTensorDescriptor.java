/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Immutable, versioned description of a quantized tensor's serialized representation.
 *
 * <p>This descriptor captures representation identity and structural layout only. It does
 * not imply that a codec or execution backend for the identified format is available.</p>
 */
public final class QuantizedTensorDescriptor {
    private static final Pattern FORMAT_ID = Pattern.compile("[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,127}");

    public enum PackingOrder {
        NOT_APPLICABLE,
        LOW_BITS_FIRST,
        HIGH_BITS_FIRST,
        FORMAT_DEFINED
    }

    private final String formatId;
    private final int formatVersion;
    private final long[] logicalShape;
    private final String logicalDtype;
    private final String storageDtype;
    private final int bitWidth;
    private final PackingOrder packingOrder;
    private final long payloadLengthBytes;
    private final Integer quantizationAxis;
    private final Integer groupSize;
    private final long scaleCount;
    private final long zeroPointCount;
    private final String codebookId;
    private final byte[] auxiliaryMetadata;

    private QuantizedTensorDescriptor(Builder b) {
        formatId = requireFormatId(b.formatId);
        if (b.formatVersion < 1) throw new IllegalArgumentException("formatVersion must be >= 1");
        formatVersion = b.formatVersion;
        logicalShape = validateShape(b.logicalShape);
        logicalDtype = requireText(b.logicalDtype, "logicalDtype");
        storageDtype = requireText(b.storageDtype, "storageDtype");
        if (b.bitWidth < 1 || b.bitWidth > 64) throw new IllegalArgumentException("bitWidth must be between 1 and 64");
        bitWidth = b.bitWidth;
        packingOrder = Objects.requireNonNull(b.packingOrder, "packingOrder");
        if (b.payloadLengthBytes < 0) throw new IllegalArgumentException("payloadLengthBytes must be >= 0");
        payloadLengthBytes = b.payloadLengthBytes;
        if (b.quantizationAxis != null && (b.quantizationAxis < 0 || b.quantizationAxis >= logicalShape.length)) {
            throw new IllegalArgumentException("quantizationAxis is outside logicalShape rank");
        }
        quantizationAxis = b.quantizationAxis;
        if (b.groupSize != null && b.groupSize < 1) throw new IllegalArgumentException("groupSize must be >= 1");
        groupSize = b.groupSize;
        if (b.scaleCount < 0 || b.zeroPointCount < 0) throw new IllegalArgumentException("metadata counts must be >= 0");
        scaleCount = b.scaleCount;
        zeroPointCount = b.zeroPointCount;
        codebookId = b.codebookId == null ? null : requireText(b.codebookId, "codebookId");
        auxiliaryMetadata = b.auxiliaryMetadata == null ? new byte[0] : b.auxiliaryMetadata.clone();

        long elements = elementCount(logicalShape);
        long requiredBytes;
        try {
            requiredBytes = Math.addExact(Math.multiplyExact(elements, (long) bitWidth), 7L) / 8L;
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("logical tensor size overflows serialized bit length", e);
        }
        if (payloadLengthBytes < requiredBytes) {
            throw new IllegalArgumentException("payloadLengthBytes is smaller than the minimum packed payload: " + requiredBytes);
        }
    }

    public static Builder builder() { return new Builder(); }

    public String getFormatId() { return formatId; }
    public int getFormatVersion() { return formatVersion; }
    public long[] getLogicalShape() { return logicalShape.clone(); }
    public String getLogicalDtype() { return logicalDtype; }
    public String getStorageDtype() { return storageDtype; }
    public int getBitWidth() { return bitWidth; }
    public PackingOrder getPackingOrder() { return packingOrder; }
    public long getPayloadLengthBytes() { return payloadLengthBytes; }
    public Integer getQuantizationAxis() { return quantizationAxis; }
    public Integer getGroupSize() { return groupSize; }
    public long getScaleCount() { return scaleCount; }
    public long getZeroPointCount() { return zeroPointCount; }
    public String getCodebookId() { return codebookId; }
    public byte[] getAuxiliaryMetadata() { return auxiliaryMetadata.clone(); }

    private static String requireFormatId(String value) {
        if (value == null || !FORMAT_ID.matcher(value).matches()) {
            throw new IllegalArgumentException("formatId must be a non-empty stable identifier (max 128 characters)");
        }
        return value;
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(field + " must not be blank");
        return value;
    }

    private static long[] validateShape(long[] shape) {
        Objects.requireNonNull(shape, "logicalShape");
        if (shape.length == 0) throw new IllegalArgumentException("logicalShape must have at least one dimension");
        long[] copy = shape.clone();
        for (long dimension : copy) {
            if (dimension < 0) throw new IllegalArgumentException("logicalShape dimensions must be >= 0");
        }
        elementCount(copy);
        return copy;
    }

    private static long elementCount(long[] shape) {
        long count = 1;
        try {
            for (long dimension : shape) count = Math.multiplyExact(count, dimension);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("logicalShape element count overflows", e);
        }
        return count;
    }

    public static final class Builder {
        private String formatId;
        private int formatVersion = 1;
        private long[] logicalShape;
        private String logicalDtype;
        private String storageDtype;
        private int bitWidth;
        private PackingOrder packingOrder = PackingOrder.FORMAT_DEFINED;
        private long payloadLengthBytes;
        private Integer quantizationAxis;
        private Integer groupSize;
        private long scaleCount;
        private long zeroPointCount;
        private String codebookId;
        private byte[] auxiliaryMetadata;

        private Builder() {}
        public Builder formatId(String value) { formatId = value; return this; }
        public Builder formatVersion(int value) { formatVersion = value; return this; }
        public Builder logicalShape(long... value) { logicalShape = value == null ? null : value.clone(); return this; }
        public Builder logicalDtype(String value) { logicalDtype = value; return this; }
        public Builder storageDtype(String value) { storageDtype = value; return this; }
        public Builder bitWidth(int value) { bitWidth = value; return this; }
        public Builder packingOrder(PackingOrder value) { packingOrder = value; return this; }
        public Builder payloadLengthBytes(long value) { payloadLengthBytes = value; return this; }
        public Builder quantizationAxis(Integer value) { quantizationAxis = value; return this; }
        public Builder groupSize(Integer value) { groupSize = value; return this; }
        public Builder scaleCount(long value) { scaleCount = value; return this; }
        public Builder zeroPointCount(long value) { zeroPointCount = value; return this; }
        public Builder codebookId(String value) { codebookId = value; return this; }
        public Builder auxiliaryMetadata(byte[] value) { auxiliaryMetadata = value == null ? null : value.clone(); return this; }
        public QuantizedTensorDescriptor build() { return new QuantizedTensorDescriptor(this); }
    }
}
