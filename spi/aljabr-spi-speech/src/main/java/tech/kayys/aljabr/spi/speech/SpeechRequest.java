/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

import tech.kayys.aljabr.spi.audio.AudioBuffer;

/** Immutable request to recognize speech from an audio buffer. */
public record SpeechRequest(AudioBuffer audio, String language, boolean timestamps) {
    public SpeechRequest {
        if (audio == null) {
            throw new IllegalArgumentException("audio must not be null");
        }
        if (language != null && language.isBlank()) {
            language = null;
        }
    }

    public static SpeechRequest of(AudioBuffer audio) {
        return new SpeechRequest(audio, null, false);
    }

    public SpeechRequest withLanguage(String language) {
        return new SpeechRequest(audio, language, timestamps);
    }

    public SpeechRequest withTimestamps() {
        return new SpeechRequest(audio, language, true);
    }
}
