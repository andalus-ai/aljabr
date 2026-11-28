package tech.kayys.aljabr.models.qwen3asr;

import java.util.List;
import java.util.Objects;

/**
 * Immutable architecture configuration for the Qwen3-ASR 1.7B-hf checkpoint.
 * This is metadata, not an executable encoder/decoder.
 */
public record Qwen3AsrConfig(AudioEncoderConfig audio, TextDecoderConfig text,
        int audioTokenId, int timestampTokenId, int padTokenId,
        List<Integer> eosTokenIds, boolean tieWordEmbeddings) {

    public Qwen3AsrConfig {
        Objects.requireNonNull(audio, "audio");
        Objects.requireNonNull(text, "text");
        eosTokenIds = List.copyOf(Objects.requireNonNull(eosTokenIds, "eosTokenIds"));
        if (audioTokenId < 0 || timestampTokenId < 0 || padTokenId < 0) {
            throw new IllegalArgumentException("special token IDs must be non-negative");
        }
        if (eosTokenIds.isEmpty() || eosTokenIds.stream().anyMatch(id -> id == null || id < 0)) {
            throw new IllegalArgumentException("eosTokenIds must contain non-negative token IDs");
        }
    }

    /** Published Qwen/Qwen3-ASR-1.7B-hf architecture values. */
    public static Qwen3AsrConfig qwen3Asr17B() {
        return new Qwen3AsrConfig(
                new AudioEncoderConfig(128, 1024, 24, 16, 16, 4096, 480, 2048, 50, 800, 500),
                new TextDecoderConfig(2048, 6144, 28, 16, 8, 128,
                        151_936, 65_536, 1.0e-6, 1_000_000.0),
                151_676, 151_705, 151_645, List.of(151_643, 151_645), true);
    }

    /** Configuration of the acoustic encoder. */
    public record AudioEncoderConfig(int numMelBins, int modelDimension, int layers,
            int attentionHeads, int keyValueHeads, int feedForwardDimension,
            int downsampleHiddenSize, int outputDimension, int windowFrames,
            int inferenceWindowFrames, int convolutionChunkSize) {
        public AudioEncoderConfig {
            positive(numMelBins, "numMelBins"); positive(modelDimension, "modelDimension");
            positive(layers, "layers"); positive(attentionHeads, "attentionHeads");
            positive(keyValueHeads, "keyValueHeads"); positive(feedForwardDimension, "feedForwardDimension");
            positive(downsampleHiddenSize, "downsampleHiddenSize"); positive(outputDimension, "outputDimension");
            positive(windowFrames, "windowFrames"); positive(inferenceWindowFrames, "inferenceWindowFrames");
            positive(convolutionChunkSize, "convolutionChunkSize");
            if (modelDimension % attentionHeads != 0) {
                throw new IllegalArgumentException("modelDimension must be divisible by attentionHeads");
            }
            if (inferenceWindowFrames % (2 * windowFrames) != 0) {
                throw new IllegalArgumentException("inferenceWindowFrames must align to two-window encoder chunks");
            }
        }
    }

    /** Text backbone configuration; implementation is shared with Qwen3. */
    public record TextDecoderConfig(int hiddenSize, int intermediateSize, int layers,
            int attentionHeads, int keyValueHeads, int headDimension, int vocabularySize,
            int maxPositionEmbeddings, double rmsNormEpsilon, double ropeTheta) {
        public TextDecoderConfig {
            positive(hiddenSize, "hiddenSize"); positive(intermediateSize, "intermediateSize");
            positive(layers, "layers"); positive(attentionHeads, "attentionHeads");
            positive(keyValueHeads, "keyValueHeads"); positive(headDimension, "headDimension");
            positive(vocabularySize, "vocabularySize"); positive(maxPositionEmbeddings, "maxPositionEmbeddings");
            if (!Double.isFinite(rmsNormEpsilon) || rmsNormEpsilon <= 0.0) {
                throw new IllegalArgumentException("rmsNormEpsilon must be finite and positive");
            }
            if (!Double.isFinite(ropeTheta) || ropeTheta <= 0.0) {
                throw new IllegalArgumentException("ropeTheta must be finite and positive");
            }
            if (hiddenSize != attentionHeads * headDimension) {
                throw new IllegalArgumentException("hiddenSize must equal attentionHeads * headDimension");
            }
            if (attentionHeads % keyValueHeads != 0) {
                throw new IllegalArgumentException("attentionHeads must be divisible by keyValueHeads");
            }
        }
    }

    private static void positive(int value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
    }
}
