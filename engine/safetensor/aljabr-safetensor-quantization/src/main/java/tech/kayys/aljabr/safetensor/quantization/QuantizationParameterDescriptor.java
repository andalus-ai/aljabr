/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;

/**
 * Serialized layout of one quantization-parameter tensor (for example, scales
 * or zero points). This describes representation metadata, not parameter values.
 */
public final class QuantizationParameterDescriptor {
    private final String logicalDtype;
    private final String storageDtype;
    private final long[] shape;
    private final int bitWidth;
    private final QuantizedTensorDescriptor.PackingOrder packingOrder;
    private final long payloadLengthBytes;

    public QuantizationParameterDescriptor(
            String logicalDtype,
            String storageDtype,
            long[] shape,
            int bitWidth,
            QuantizedTensorDescriptor.PackingOrder packingOrder,
            long payloadLengthBytes) {
        this.logicalDtype = requireText(logicalDtype, "logicalDtype");
        this.storageDtype = requireText(storageDtype, "storageDtype");
        this.shape = validateShape(shape);
        if (bitWidth < 1 || bitWidth > 64) {
            throw new IllegalArgumentException("bitWidth must be between 1 and 64");
        }
        this.bitWidth = bitWidth;
        this.packingOrder = Objects.requireNonNull(packingOrder, "packingOrder");
        if (payloadLengthBytes < 0) {
            throw new IllegalArgumentException("payloadLengthBytes must be >= 0");
        }
        this.payloadLengthBytes = payloadLengthBytes;
        long requiredBytes = minimumPayloadBytes(this.shape, bitWidth);
        if (payloadLengthBytes < requiredBytes) {
            throw new IllegalArgumentException(
                    "payloadLengthBytes is smaller than the minimum packed parameter payload: " + requiredBytes);
        }
    }

    public String getLogicalDtype() { return logicalDtype; }
    public String getStorageDtype() { return storageDtype; }
    public long[] getShape() { return shape.clone(); }
    public int getBitWidth() { return bitWidth; }
    public QuantizedTensorDescriptor.PackingOrder getPackingOrder() { return packingOrder; }
    public long getPayloadLengthBytes() { return payloadLengthBytes; }

    /** Returns the checked element count represented by this shape. */
    public long getElementCount() {
        long count = 1;
        try {
            for (long dimension : shape) count = Math.multiplyExact(count, dimension);
        } catch (ArithmeticException e) {
            throw new IllegalStateException("validated parameter shape overflowed", e);
        }
        return count;
    }

    private static long[] validateShape(long[] value) {
        Objects.requireNonNull(value, "shape");
        if (value.length == 0) throw new IllegalArgumentException("shape must have at least one dimension");
        long[] copy = value.clone();
        long count = 1;
        try {
            for (long dimension : copy) {
                if (dimension < 0) throw new IllegalArgumentException("shape dimensions must be >= 0");
                count = Math.multiplyExact(count, dimension);
            }
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("parameter shape element count overflows", e);
        }
        return copy;
    }

    private static long minimumPayloadBytes(long[] shape, int bitWidth) {
        long count = 1;
        try {
            for (long dimension : shape) count = Math.multiplyExact(count, dimension);
            return Math.addExact(Math.multiplyExact(count, (long) bitWidth), 7L) / 8L;
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("parameter payload bit length overflows", e);
        }
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(field + " must not be blank");
        return value;
    }
}
