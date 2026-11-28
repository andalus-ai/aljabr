/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

/** A recognized speech segment with optional timing information in seconds. */
public record SpeechSegment(String text, double startSeconds, double endSeconds) {
    public SpeechSegment {
        if (text == null) {
            throw new IllegalArgumentException("text must not be null");
        }
        boolean startMissing = Double.isNaN(startSeconds);
        boolean endMissing = Double.isNaN(endSeconds);
        if (startMissing != endMissing) {
            throw new IllegalArgumentException("both timestamps must be present or both must be NaN");
        }
        if (!startMissing && (!Double.isFinite(startSeconds) || !Double.isFinite(endSeconds))) {
            throw new IllegalArgumentException("timestamps must be finite");
        }
        if (!startMissing && (startSeconds < 0 || endSeconds < startSeconds)) {
            throw new IllegalArgumentException("timestamps must be non-negative and end must not precede start");
        }
    }

    public static SpeechSegment untimed(String text) {
        return new SpeechSegment(text, Double.NaN, Double.NaN);
    }

    public boolean hasTimestamps() {
        return !Double.isNaN(startSeconds) && !Double.isNaN(endSeconds);
    }
}
