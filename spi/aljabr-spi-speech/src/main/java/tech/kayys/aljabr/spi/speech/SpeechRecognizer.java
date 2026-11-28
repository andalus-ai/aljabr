/*
 * Aljabr — Speech SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.speech;

/**
 * Model-neutral speech recognition capability.
 *
 * <p>Implementations may be backed by encoder-decoder, CTC, transducer, or
 * audio-language-model architectures. The contract does not prescribe a
 * decoding strategy or serving/streaming policy.</p>
 */
public interface SpeechRecognizer extends AutoCloseable {
    SpeechResult transcribe(SpeechRequest request);

    @Override
    void close();
}
