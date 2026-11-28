/*
 * Aljabr — Audio SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.audio;

/**
 * Pull-based source of sequential audio buffers.
 *
 * <p>Each non-null buffer returned by {@link #read()} is owned by the caller
 * and must be closed by it. A null result signals end-of-stream. Implementors
 * must document thread-safety; the default contract is single-reader.</p>
 */
public interface AudioSource extends AutoCloseable {
    AudioFormat format();

    /** Reads the next chunk, or returns null when the source is exhausted. */
    AudioBuffer read();

    @Override
    void close();
}
