package tech.kayys.aljabr.models.qwen3asr.encoder;

import tech.kayys.aljabr.models.qwen3asr.Qwen3AsrConfig;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable execution geometry for Qwen3-ASR's chunked acoustic encoder.
 *
 * <p>This planner deliberately contains no tensor kernels or model weights. It
 * translates a Mel-frame length into independently convolved chunks and
 * block-attention windows so the eventual encoder executor can schedule work
 * without reimplementing checkpoint-specific geometry.</p>
 */
public final class Qwen3AsrEncoderPlan {
    private static final int CONV_STAGES = 3;

    private final Qwen3AsrConfig.AudioEncoderConfig config;

    public Qwen3AsrEncoderPlan(Qwen3AsrConfig.AudioEncoderConfig config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Plans the encoder for one unpadded Mel sequence.
     *
     * <p>Convolution runs independently on chunks of {@code 2 * windowFrames}
     * Mel frames. Attention groups the resulting chunk embeddings into
     * inference windows. Positional indices restart at zero for every
     * convolution chunk, matching the reference implementation.</p>
     */
    public Plan plan(int melFrameCount) {
        if (melFrameCount <= 0) {
            throw new IllegalArgumentException("melFrameCount must be positive");
        }

        int melFramesPerChunk = Math.multiplyExact(config.windowFrames(), 2);
        int chunksPerAttentionWindow =
                config.inferenceWindowFrames() / melFramesPerChunk;

        List<ConvolutionChunk> chunks = new ArrayList<>(
                ceilDiv(melFrameCount, melFramesPerChunk));
        int inputStart = 0;
        int totalTokens = 0;
        int chunkIndex = 0;
        while (inputStart < melFrameCount) {
            int frameCount = Math.min(melFramesPerChunk, melFrameCount - inputStart);
            int outputTokens = convolutionOutputLength(frameCount);
            int attentionWindow = chunkIndex / chunksPerAttentionWindow;
            chunks.add(new ConvolutionChunk(
                    chunkIndex, inputStart, frameCount, outputTokens,
                    0, attentionWindow));
            totalTokens = Math.addExact(totalTokens, outputTokens);
            inputStart = Math.addExact(inputStart, frameCount);
            chunkIndex++;
        }

        List<AttentionWindow> windows = new ArrayList<>(
                ceilDiv(chunks.size(), chunksPerAttentionWindow));
        for (int firstChunk = 0, windowIndex = 0;
                firstChunk < chunks.size();
                firstChunk += chunksPerAttentionWindow, windowIndex++) {
            int endChunk = Math.min(chunks.size(), firstChunk + chunksPerAttentionWindow);
            int firstToken = 0;
            for (int i = 0; i < firstChunk; i++) {
                firstToken = Math.addExact(firstToken, chunks.get(i).outputTokenCount());
            }
            int tokenCount = 0;
            for (int i = firstChunk; i < endChunk; i++) {
                tokenCount = Math.addExact(tokenCount, chunks.get(i).outputTokenCount());
            }
            windows.add(new AttentionWindow(windowIndex, firstChunk,
                    endChunk - firstChunk, firstToken, tokenCount));
        }

        return new Plan(melFrameCount, totalTokens, chunks, windows);
    }

    /** Three stride-2, kernel-3, padding-1 convolutions, using ceil division. */
    public static int convolutionOutputLength(int inputLength) {
        if (inputLength <= 0) {
            throw new IllegalArgumentException("inputLength must be positive");
        }
        int length = inputLength;
        for (int stage = 0; stage < CONV_STAGES; stage++) {
            length = (length + 1) / 2;
        }
        return length;
    }

    private static int ceilDiv(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    public record ConvolutionChunk(
            int index,
            int inputStartFrame,
            int inputFrameCount,
            int outputTokenCount,
            int positionalOffset,
            int attentionWindowIndex) {
        public ConvolutionChunk {
            if (index < 0 || inputStartFrame < 0 || inputFrameCount <= 0
                    || outputTokenCount <= 0 || positionalOffset != 0
                    || attentionWindowIndex < 0) {
                throw new IllegalArgumentException("invalid convolution chunk geometry");
            }
        }
    }

    public record AttentionWindow(
            int index,
            int firstChunkIndex,
            int chunkCount,
            int firstTokenIndex,
            int tokenCount) {
        public AttentionWindow {
            if (index < 0 || firstChunkIndex < 0 || chunkCount <= 0
                    || firstTokenIndex < 0 || tokenCount <= 0) {
                throw new IllegalArgumentException("invalid attention window geometry");
            }
        }
    }

    public record Plan(
            int melFrameCount,
            int totalOutputTokens,
            List<ConvolutionChunk> convolutionChunks,
            List<AttentionWindow> attentionWindows) {
        public Plan {
            if (melFrameCount <= 0 || totalOutputTokens <= 0) {
                throw new IllegalArgumentException("plan lengths must be positive");
            }
            convolutionChunks = List.copyOf(convolutionChunks);
            attentionWindows = List.copyOf(attentionWindows);
        }
    }
}
