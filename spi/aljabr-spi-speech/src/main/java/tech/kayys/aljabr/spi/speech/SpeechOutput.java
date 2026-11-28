/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

import tech.kayys.aljabr.spi.audio.AudioBuffer;

/** Generated speech audio. The caller owns and must close the returned buffer. */
public record SpeechOutput(AudioBuffer audio, String voice) implements AutoCloseable {
    public SpeechOutput {
        if (audio == null) {
            throw new IllegalArgumentException("audio must not be null");
        }
    }

    @Override
    public void close() {
        audio.close();
    }
}
