/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

/** Immutable text-to-speech request. */
public record SpeechSynthesisRequest(String text, String language, String voice) {
    public SpeechSynthesisRequest {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("text must not be blank");
        }
        if (language != null && language.isBlank()) {
            language = null;
        }
        if (voice != null && voice.isBlank()) {
            voice = null;
        }
    }

    public static SpeechSynthesisRequest of(String text) {
        return new SpeechSynthesisRequest(text, null, null);
    }
}
