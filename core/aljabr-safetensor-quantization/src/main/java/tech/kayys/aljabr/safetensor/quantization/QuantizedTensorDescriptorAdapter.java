/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;

/**
 * Explicit compatibility bridge from legacy {@link QuantizedTensorInfo} metadata
 * to the versioned serialized-representation descriptor.
 *
 * <p>Legacy metadata does not identify a serialization format, format version,
 * storage dtype, bit width, packing order, or payload length. Those values must
 * therefore be supplied by the caller; this adapter never guesses them from
 * dtype labels or quantization strategy names.</p>
 */
public final class QuantizedTensorDescriptorAdapter {

    private QuantizedTensorDescriptorAdapter() {
    }

    /**
     * Converts legacy metadata when all representation-critical fields are known.
     *
     * @param info legacy quantized tensor metadata
     * @param representation explicit serialized representation details
     * @return immutable versioned descriptor
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if required legacy metadata is absent or
     *         the resulting descriptor is structurally invalid
     */
    public static QuantizedTensorDescriptor toDescriptor(
            QuantizedTensorInfo info,
            Representation representation) {
        Objects.requireNonNull(info, "info");
        Objects.requireNonNull(representation, "representation");

        int[] legacyShape = Objects.requireNonNull(info.getShape(), "info.shape");
        long[] shape = new long[legacyShape.length];
        for (int i = 0; i < legacyShape.length; i++) {
            shape[i] = legacyShape[i];
        }

        float[] scales = info.getScales();
        float[] zeroPoints = info.getZeros();
        byte[] metadata = info.getMetadata();

        return QuantizedTensorDescriptor.builder()
                .formatId(representation.formatId)
                .formatVersion(representation.formatVersion)
                .logicalShape(shape)
                .logicalDtype(info.getOriginalDtype())
                .storageDtype(representation.storageDtype)
                .bitWidth(representation.bitWidth)
                .packingOrder(representation.packingOrder)
                .payloadLengthBytes(representation.payloadLengthBytes)
                .quantizationAxis(representation.quantizationAxis)
                .groupSize(representation.groupSize)
                .scaleCount(scales == null ? 0 : scales.length)
                .zeroPointCount(zeroPoints == null ? 0 : zeroPoints.length)
                .auxiliaryMetadata(metadata)
                .build();
    }

    /**
     * Caller-supplied facts about the serialized representation. No defaults are
     * provided because a default could falsely claim wire-format compatibility.
     */
    public static final class Representation {
        private final String formatId;
        private final int formatVersion;
        private final String storageDtype;
        private final int bitWidth;
        private final QuantizedTensorDescriptor.PackingOrder packingOrder;
        private final long payloadLengthBytes;
        private final Integer quantizationAxis;
        private final Integer groupSize;

        public Representation(
                String formatId,
                int formatVersion,
                String storageDtype,
                int bitWidth,
                QuantizedTensorDescriptor.PackingOrder packingOrder,
                long payloadLengthBytes,
                Integer quantizationAxis,
                Integer groupSize) {
            this.formatId = Objects.requireNonNull(formatId, "formatId");
            this.formatVersion = formatVersion;
            this.storageDtype = Objects.requireNonNull(storageDtype, "storageDtype");
            this.bitWidth = bitWidth;
            this.packingOrder = Objects.requireNonNull(packingOrder, "packingOrder");
            this.payloadLengthBytes = payloadLengthBytes;
            this.quantizationAxis = quantizationAxis;
            this.groupSize = groupSize;
        }
    }
}
