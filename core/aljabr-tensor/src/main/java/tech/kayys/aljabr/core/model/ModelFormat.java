package tech.kayys.aljabr.core.model;

import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Enumeration of supported model formats.
 * @author bhangun
 */
public enum ModelFormat {

        // Convertible formats (can be converted to GGUF)
        PYTORCH("pytorch", "PyTorch",
                        Set.of(".bin", ".pt", ".pth"),
                        Set.of("pytorch_model.bin", "model.safetensors"),
                        true,
                        "PyTorch"),

        SAFETENSORS("safetensors", "SafeTensors",
                        Set.of(".safetensors"),
                        Set.of("model.safetensors"),
                        true,
                        "SafeTensors"),

        TENSORFLOW("tensorflow", "TensorFlow",
                        Set.of(".pb", ".h5"),
                        Set.of("saved_model.pb", "tf_model.h5"),
                        true,
                        "TensorFlow"),

        FLAX("flax", "Flax/JAX",
                        Set.of(".msgpack"),
                        Set.of("flax_model.msgpack"),
                        true,
                        "JAX"),

        // Native inference formats (no conversion needed)
        GGUF("gguf", "GGUF",
                        Set.of(".gguf"),
                        Set.of(),
                        false,
                        "llama.cpp"),

        LITERT("litert", "LiteRT",
                        Set.of(".litertlm"),
                        Set.of(),
                        false,
                        "LiteRT"),

        ONNX("onnx", "ONNX",
                        Set.of(".onnx"),
                        Set.of("model.onnx"),
                        false,
                        "ONNX Runtime"),

        TENSORRT("trt", "TensorRT",
                        Set.of(".trt", ".engine"),
                        Set.of(),
                        false,
                        "TensorRT"),

        TORCHSCRIPT("torchscript", "TorchScript",
                        Set.of(".pt", ".pts"),
                        Set.of(),
                        false,
                        "PyTorch"),

        TENSORFLOW_SAVED_MODEL("pb", "TensorFlow SavedModel",
                        Set.of(".pb"),
                        Set.of("saved_model.pb"),
                        false,
                        "TensorFlow"),

        UNKNOWN("unknown", "Unknown",
                        Set.of(),
                        Set.of(),
                        false,
                        "Unknown");

        private final String id;
        private final String displayName;
        private final Set<String> fileExtensions;
        private final Set<String> markerFiles;
        private final boolean requiresConversion;
        private final String runtime;

        ModelFormat(String id, String displayName,
                        Set<String> fileExtensions, Set<String> markerFiles,
                        boolean requiresConversion, String runtime) {
                this.id = id;
                this.displayName = displayName;
                this.fileExtensions = Collections.unmodifiableSet(fileExtensions);
                this.markerFiles = Collections.unmodifiableSet(markerFiles);
                this.requiresConversion = requiresConversion;
                this.runtime = runtime;
        }

        public String getId() {
                return id;
        }

        public String getDisplayName() {
                return displayName;
        }

        public Set<String> getFileExtensions() {
                return fileExtensions;
        }

        public String getExtension() {
                return fileExtensions.isEmpty() ? "" : fileExtensions.iterator().next();
        }

        public Set<String> getMarkerFiles() {
                return markerFiles;
        }

        public String getRuntime() {
                return runtime;
        }

        public boolean isRequiresConversion() {
                return requiresConversion;
        }

        public boolean requiresConversion() {
                return requiresConversion;
        }

        public boolean isConvertible() {
                return requiresConversion && this != UNKNOWN;
        }

        public static ModelFormat fromId(String id) {
                if (id == null || id.isBlank())
                        return UNKNOWN;
                String normalized = id.toLowerCase().trim();
                return Arrays.stream(values())
                                .filter(f -> f.id.equals(normalized))
                                .findFirst()
                                .orElse(UNKNOWN);
        }

        public static ModelFormat fromExtension(String extension) {
                if (extension == null || extension.isBlank())
                        return UNKNOWN;
                String normalized = extension.toLowerCase().trim();
                if (!normalized.startsWith("."))
                        normalized = "." + normalized;
                final String ext = normalized;
                return Arrays.stream(values())
                                .filter(f -> f.fileExtensions.contains(ext))
                                .findFirst()
                                .orElse(UNKNOWN);
        }

        public static Optional<ModelFormat> findByExtension(String extension) {
                ModelFormat format = fromExtension(extension);
                return format == UNKNOWN ? Optional.empty() : Optional.of(format);
        }

        public static Optional<ModelFormat> findByRuntime(String runtime) {
                if (runtime == null || runtime.isBlank())
                        return Optional.empty();
                return Arrays.stream(values())
                                .filter(format -> format.runtime.equalsIgnoreCase(runtime.trim()))
                                .findFirst();
        }

        public static ModelFormat fromMarkerFile(String filename) {
                if (filename == null || filename.isBlank())
                        return UNKNOWN;
                String normalized = filename.toLowerCase().trim();
                return Arrays.stream(values())
                                .filter(f -> f.markerFiles.stream()
                                                .anyMatch(marker -> marker.equalsIgnoreCase(normalized)))
                                .findFirst()
                                .orElse(UNKNOWN);
        }

        public static Set<ModelFormat> getConvertibleFormats() {
                return Arrays.stream(values()).filter(ModelFormat::isRequiresConversion)
                                .collect(Collectors.toUnmodifiableSet());
        }

        public static Set<ModelFormat> getNativeFormats() {
                return Arrays.stream(values()).filter(f -> !f.requiresConversion && f != UNKNOWN)
                                .collect(Collectors.toUnmodifiableSet());
        }

        public boolean matches(String filename) {
                if (filename == null || filename.isBlank())
                        return false;
                String lower = filename.toLowerCase();
                return fileExtensions.stream().anyMatch(lower::endsWith);
        }

        @Override
        public String toString() {
                return String.format("%s (%s) - Runtime: %s, Convertible: %s", displayName, id, runtime,
                                requiresConversion);
        }
}
