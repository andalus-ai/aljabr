/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.loader;

/**
 * Resource limits applied while validating SafeTensors headers.
 *
 * <p>Defaults are intentionally generous for real-world model checkpoints.
 * Applications processing untrusted files can supply stricter limits.
 */
public record SafetensorParserLimits(
        long maxHeaderBytes,
        long maxTensorCount,
        int maxMetadataEntries,
        int maxMetadataStringLength,
        int maxTensorRank,
        int maxTensorNameLength) {

    private static final long MIB = 1024L * 1024L;

    public SafetensorParserLimits {
        if (maxHeaderBytes < 2) {
            throw new IllegalArgumentException("maxHeaderBytes must be >= 2");
        }
        if (maxTensorCount < 1) {
            throw new IllegalArgumentException("maxTensorCount must be >= 1");
        }
        if (maxMetadataEntries < 0) {
            throw new IllegalArgumentException("maxMetadataEntries must be >= 0");
        }
        if (maxMetadataStringLength < 1) {
            throw new IllegalArgumentException("maxMetadataStringLength must be >= 1");
        }
        if (maxTensorRank < 0) {
            throw new IllegalArgumentException("maxTensorRank must be >= 0");
        }
        if (maxTensorNameLength < 1) {
            throw new IllegalArgumentException("maxTensorNameLength must be >= 1");
        }
    }

    /** Defaults support large models while bounding pathological header complexity. */
    public static SafetensorParserLimits defaults() {
        return new SafetensorParserLimits(
                100L * MIB,
                1_000_000L,
                100_000,
                1_048_576,
                64,
                16_384);
    }
}
