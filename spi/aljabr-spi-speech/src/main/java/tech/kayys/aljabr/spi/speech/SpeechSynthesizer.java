/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

/** Model-neutral text-to-speech capability. */
public interface SpeechSynthesizer extends AutoCloseable {
    SpeechOutput synthesize(SpeechSynthesisRequest request);

    @Override
    void close();
}
