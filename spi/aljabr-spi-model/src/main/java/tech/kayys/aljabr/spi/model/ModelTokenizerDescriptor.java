package tech.kayys.aljabr.spi.model;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Lightweight tokenizer profile for a model-family extension.
 *
 * <p>
 * A descriptor names the tokenizer algorithm and the file groups that make a
 * usable tokenizer artifact. Each inner list is an alternative "all files must
 * exist" group, which lets families support both {@code tokenizer.json} and
 * split {@code vocab.json + merges.txt} layouts without shipping code.
 * </p>
 * @author bhangun
 */
public record ModelTokenizerDescriptor(
        String id,
        ModelTokenizerKind kind,
        List<List<String>> requiredFileGroups,
        Map<String, String> options) {

    public ModelTokenizerDescriptor {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Tokenizer descriptor id must not be blank");
        }
        kind = kind == null ? ModelTokenizerKind.CUSTOM : kind;
        requiredFileGroups = requiredFileGroups == null
                ? List.of()
                : requiredFileGroups.stream()
                        .map(group -> group == null ? List.<String>of() : List.copyOf(group))
                        .toList();
        options = options == null ? Map.of() : Map.copyOf(options);
    }

    public static ModelTokenizerDescriptor huggingFaceBpe(String id) {
        return new ModelTokenizerDescriptor(
                id,
                ModelTokenizerKind.HUGGING_FACE_BPE,
                List.of(
                        List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json"),
                        List.of("vocab.json", "merges.txt"),
                        List.of("tokenizer/vocab.json", "tokenizer/merges.txt")),
                Map.of("pre_tokenizer", "auto"));
    }

    public static ModelTokenizerDescriptor sentencePieceBpe(String id) {
        return new ModelTokenizerDescriptor(
                id,
                ModelTokenizerKind.SENTENCE_PIECE_BPE,
                List.of(
                        List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "native_sentencepiece", "unsupported"));
    }

    public static ModelTokenizerDescriptor wordPiece(String id) {
        return new ModelTokenizerDescriptor(
                id,
                ModelTokenizerKind.WORD_PIECE,
                List.of(
                        List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json"),
                        List.of("vocab.txt"),
                        List.of("tokenizer/vocab.txt")),
                Map.of("pre_tokenizer", "bert-basic"));
    }

    public Optional<List<Path>> firstExistingFileGroup(Path modelDir) {
        if (modelDir == null) {
            return Optional.empty();
        }
        for (List<String> group : requiredFileGroups) {
            if (group.isEmpty()) {
                continue;
            }
            List<Path> resolved = group.stream()
                    .map(modelDir::resolve)
                    .toList();
            if (resolved.stream().allMatch(Files::exists)) {
                return Optional.of(resolved);
            }
        }
        return Optional.empty();
    }

    public boolean isStrict() {
        return Boolean.parseBoolean(options.getOrDefault("strict", "false"));
    }
}
