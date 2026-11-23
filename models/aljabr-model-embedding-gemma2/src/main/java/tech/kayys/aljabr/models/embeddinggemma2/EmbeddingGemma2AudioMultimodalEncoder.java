package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.audio.EmbeddingGemma2AudioEncoder;
import tech.kayys.aljabr.models.embeddinggemma2.audio.EmbeddingGemma2AudioProcessor;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2MultimodalEmbedder;

/**
 * End-to-end waveform-and-text encoder for EmbeddingGemma 2 audio inputs.
 *
 * <p>PCM decoding/container I/O and prompt expansion remain caller-owned.
 * The caller supplies an already-expanded text sequence with one configured
 * audio placeholder for every valid audio soft token produced by the audio
 * tower. This composition performs waveform preprocessing, audio encoding,
 * modality-to-text projection, placeholder assembly, and text encoding.</p>
 */
public final class EmbeddingGemma2AudioMultimodalEncoder {
    private final EmbeddingGemma2TextBackbone textBackbone;
    private final EmbeddingGemma2AudioProcessor audioProcessor;
    private final EmbeddingGemma2AudioEncoder audioEncoder;
    private final EmbeddingGemma2MultimodalEmbedder audioEmbedder;
    private final int audioTokenId;
    private final int padTokenId;

    public EmbeddingGemma2AudioMultimodalEncoder(
            EmbeddingGemma2TextBackbone textBackbone,
            EmbeddingGemma2AudioProcessor audioProcessor,
            EmbeddingGemma2AudioEncoder audioEncoder,
            EmbeddingGemma2MultimodalEmbedder audioEmbedder,
            int audioTokenId,
            int padTokenId) {
        this.textBackbone = Objects.requireNonNull(textBackbone, "textBackbone must not be null");
        this.audioProcessor = Objects.requireNonNull(audioProcessor, "audioProcessor must not be null");
        this.audioEncoder = Objects.requireNonNull(audioEncoder, "audioEncoder must not be null");
        this.audioEmbedder = Objects.requireNonNull(audioEmbedder, "audioEmbedder must not be null");
        if (audioTokenId < 0 || padTokenId < 0) {
            throw new IllegalArgumentException("audioTokenId and padTokenId must be non-negative");
        }
        EmbeddingGemma2AudioConfig config = audioProcessor.config();
        if (audioTokenId != config.audioTokenId()) {
            throw new IllegalArgumentException("audioTokenId must match the audio processor configuration");
        }
        if (audioEmbedder.multimodalHiddenSize() != config.outputProjectionSize()) {
            throw new IllegalArgumentException("audio bridge input size must match audio output projection size");
        }
        this.audioTokenId = audioTokenId;
        this.padTokenId = padTokenId;
    }

    /**
     * Encodes one decoded PCM waveform with an already-expanded text prompt.
     *
     * @param inputIds text token IDs including one audio placeholder per valid audio soft token
     * @param interleavedPcm frame-major decoded float PCM
     * @param channels source channel count
     * @param sourceSampleRate source PCM sample rate
     * @param validTokens optional true=valid text-token mask
     * @param positionIds explicit positions for the expanded sequence
     * @return contextualized per-token text-space states
     */
    public float[][] encode(
            int[] inputIds,
            float[] interleavedPcm,
            int channels,
            int sourceSampleRate,
            boolean[] validTokens,
            int[] positionIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        EmbeddingGemma2AudioProcessor.ProcessedAudio processed =
                audioProcessor.process(interleavedPcm, channels, sourceSampleRate);
        return encodeProcessed(inputIds, processed, validTokens, positionIds);
    }

    /**
     * Convenience path: expands one audio marker using the processor-owned
     * feature-mask policy and generates sequential text positions.
     */
    public float[][] encode(int[] inputIds, float[] interleavedPcm, int channels, int sourceSampleRate) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        EmbeddingGemma2AudioProcessor.ProcessedAudio processed =
                audioProcessor.process(interleavedPcm, channels, sourceSampleRate);
        int[] expandedIds = audioProcessor.expandAudioPlaceholders(inputIds, processed);
        int[] positions = new int[expandedIds.length];
        for (int i = 0; i < positions.length; i++) positions[i] = i;
        return encodeProcessed(expandedIds, processed, null, positions);
    }

    private float[][] encodeProcessed(
            int[] inputIds,
            EmbeddingGemma2AudioProcessor.ProcessedAudio processed,
            boolean[] validTokens,
            int[] positionIds) {
        int expectedAudioTokens = audioProcessor.audioPlaceholderCount(processed);
        EmbeddingGemma2AudioFeatureExtractor.Features features = processed.features();
        float[][][] batchFeatures = {features.inputFeatures()};
        boolean[][] batchMask = {features.inputFeaturesMask()};
        EmbeddingGemma2AudioEncoder.Output towerOutput = audioEncoder.compute(batchFeatures, batchMask);
        float[][] audioStates = selectValidStates(
                towerOutput.lastHiddenState()[0], towerOutput.attentionMask()[0]);
        if (audioStates.length != expectedAudioTokens) {
            throw new IllegalStateException("audio tower produced " + audioStates.length
                    + " valid soft tokens, but processor policy requires " + expectedAudioTokens
                    + "; verify audio subsampling and processor configuration");
        }
        int actualPlaceholders = 0;
        for (int tokenId : inputIds) if (tokenId == audioTokenId) actualPlaceholders++;
        if (actualPlaceholders != expectedAudioTokens) {
            throw new IllegalArgumentException("expanded prompt must contain " + expectedAudioTokens
                    + " audio placeholders; found " + actualPlaceholders);
        }
        float[][] projectedAudioStates = audioEmbedder.project(audioStates);

        float[][] textEmbeddings = EmbeddingGemma2MultimodalInputAssembler.embedTextTokens(
                textBackbone, inputIds, audioTokenId, padTokenId);
        float[][] mergedEmbeddings = EmbeddingGemma2MultimodalInputAssembler.replaceModalityPlaceholders(
                textEmbeddings, inputIds, audioTokenId, List.<float[][]>of(projectedAudioStates), "audio");
        return textBackbone.encodeEmbeddings(mergedEmbeddings, validTokens, positionIds);
    }

    private static float[][] selectValidStates(float[][] states, boolean[] mask) {
        if (states == null || mask == null || states.length != mask.length) {
            throw new IllegalArgumentException("audio tower states and mask must have equal lengths");
        }
        int validCount = 0;
        for (boolean valid : mask) if (valid) validCount++;
        if (validCount == 0) {
            throw new IllegalArgumentException("audio input produced no valid soft tokens");
        }
        float[][] selected = new float[validCount][];
        int output = 0;
        for (int token = 0; token < states.length; token++) {
            if (mask[token]) selected[output++] = states[token].clone();
        }
        return selected;
    }
}
