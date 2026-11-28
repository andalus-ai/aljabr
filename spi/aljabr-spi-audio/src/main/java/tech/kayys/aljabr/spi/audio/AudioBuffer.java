/*
 * Aljabr — Audio SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.audio;

import java.nio.FloatBuffer;

/**
 * A closeable, interleaved PCM audio buffer.
 *
 * <p>The returned sample view is read-only and remains valid only while this
 * buffer is open. Samples are normalized floats, conventionally in [-1, 1].
 * Implementations must document whether storage is heap- or native-backed.
 * Callers must not retain the view after closing this buffer.</p>
 */
public interface AudioBuffer extends AutoCloseable {
    AudioFormat format();

    /** Number of audio frames; one frame contains {@code format().channels()} samples. */
    long frameCount();

    /**
     * Returns a read-only view of interleaved samples.
     *
     * <p>The view's position is zero and its limit is the number of samples.
     * The view is borrowed; ownership remains with this buffer.</p>
     */
    FloatBuffer samples();

    boolean isClosed();

    @Override
    void close();
}
