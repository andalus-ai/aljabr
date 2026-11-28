/*
 * Aljabr — Audio Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.audio;

import tech.kayys.aljabr.spi.audio.AudioBuffer;
import tech.kayys.aljabr.spi.audio.AudioFormat;

import java.nio.FloatBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Heap-backed immutable audio buffer.
 *
 * <p>Input samples are copied at construction. The returned FloatBuffer is a
 * read-only borrowed view and becomes invalid by contract when this buffer is
 * closed. This implementation is the portable reference; native/FFM storage
 * belongs in an optional provider with the same ownership semantics.</p>
 */
public final class HeapAudioBuffer implements AudioBuffer {
    private final AudioFormat format;
    private final float[] samples;
    private final AtomicBoolean closed = new AtomicBoolean();

    private HeapAudioBuffer(AudioFormat format, float[] samples) {
        this.format = Objects.requireNonNull(format, "format");
        this.samples = samples;
    }

    public static HeapAudioBuffer copyOf(AudioFormat format, float[] interleavedSamples) {
        Objects.requireNonNull(interleavedSamples, "interleavedSamples");
        if (interleavedSamples.length % format.channels() != 0) {
            throw new IllegalArgumentException("sample count must be divisible by channel count");
        }
        return new HeapAudioBuffer(format, interleavedSamples.clone());
    }

    @Override
    public AudioFormat format() {
        ensureOpen();
        return format;
    }

    @Override
    public long frameCount() {
        ensureOpen();
        return samples.length / format.channels();
    }

    @Override
    public FloatBuffer samples() {
        ensureOpen();
        return FloatBuffer.wrap(samples).asReadOnlyBuffer();
    }

    @Override
    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        closed.set(true);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("AudioBuffer is closed");
        }
    }
}
