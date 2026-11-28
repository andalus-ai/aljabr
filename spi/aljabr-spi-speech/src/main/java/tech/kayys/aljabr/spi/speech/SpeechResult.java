/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

import java.util.List;

/** Result of an ASR/STT operation. */
public record SpeechResult(String text, String detectedLanguage, List<SpeechSegment> segments) {
    public SpeechResult {
        if (text == null) {
            throw new IllegalArgumentException("text must not be null");
        }
        segments = segments == null ? List.of() : List.copyOf(segments);
    }

    public static SpeechResult text(String text) {
        return new SpeechResult(text, null, List.of());
    }
}
