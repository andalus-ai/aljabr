package tech.kayys.aljabr.ml.export;

import tech.kayys.aljabr.ml.autograd.GradTensor;
import tech.kayys.aljabr.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Unified model exporter for deployment.
 *
 * <p>
 * Supports exporting trained models to various formats for production
 * deployment:
 * </p>
 * <ul>
 * <li><b>ONNX</b> - Open Neural Network Exchange (cross-platform)</li>
 * <li><b>GGUF</b> - GPT-Generated Unified Format (llama.cpp)</li>
 * <li><b>LiteRT</b> - Google LiteRT (edge devices)</li>
 * </ul>
 *
 * <h3>Example Usage</h3>
 * 
 * <pre>{@code
 * ModelExporter exporter = ModelExporter.builder()
 *         .model(model)
 *         .inputShape(1, 3, 224, 224)
 *         .build();
 *
 * // Export to different formats
 * exporter.toONNX("model.onnx");
 * exporter.toGGUF("model.gguf", Quantization.INT4);
 * exporter.toLiteRT("model.litert");
 * }</pre>
 *
 * @author Aljabr Team
 * @version 0.1.0
 */
public class ModelExporter {

    private final NNModule model;
    private final long[] inputShape;
    private final Map<String, Object> metadata;

    private ModelExporter(Builder builder) {
        this.model = builder.model;
        this.inputShape = builder.inputShape;
        this.metadata = builder.metadata;
    }

    /**
     * Create a new builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Export model to ONNX format.
     *
     * @param outputPath path to save ONNX model
     * @throws IOException if export fails
     */
    public void toONNX(Path outputPath) throws IOException {
        tech.kayys.aljabr.ml.export.onnx.OnnxExporter.fromModel(model, inputShape).export(outputPath);
    }

    public void toONNX(String outputPath) throws IOException {
        toONNX(Path.of(outputPath));
    }

    public void toGGUF(Path outputPath, Quantization quantization) throws IOException {
        tech.kayys.aljabr.ml.export.gguf.GgufExporter.fromModel(model, metadata)
                .quantization(toGgufQuant(quantization))
                .export(outputPath);
    }

    public void toGGUF(String outputPath, Quantization quantization) throws IOException {
        toGGUF(Path.of(outputPath), quantization);
    }

    public void toLiteRT(Path outputPath) throws IOException {
        tech.kayys.aljabr.ml.export.litert.LiteRTExporter.fromModel(model, inputShape).export(outputPath);
    }

    public void toLiteRT(String outputPath) throws IOException {
        toLiteRT(Path.of(outputPath));
    }

    private static tech.kayys.aljabr.ml.export.gguf.GgufExporter.Quantization toGgufQuant(Quantization q) {
        return switch (q) {
            case FP16 -> tech.kayys.aljabr.ml.export.gguf.GgufExporter.Quantization.FP16;
            case INT8 -> tech.kayys.aljabr.ml.export.gguf.GgufExporter.Quantization.INT8;
            case INT4 -> tech.kayys.aljabr.ml.export.gguf.GgufExporter.Quantization.INT4;
            case NF4 -> tech.kayys.aljabr.ml.export.gguf.GgufExporter.Quantization.NF4;
            default -> tech.kayys.aljabr.ml.export.gguf.GgufExporter.Quantization.NONE;
        };
    }

    /**
     * Get model size estimate in bytes.
     *
     * @return estimated size
     */
    public long estimateSize() {
        if (model instanceof tech.kayys.aljabr.ml.nn.NNModule module) {
            // float32: 4 bytes per parameter
            return module.parameterCount() * 4L;
        }
        return 0L;
    }

    /**
     * Quantization methods for model compression.
     */
    public enum Quantization {
        NONE,
        FP16,
        INT8,
        INT4,
        NF4
    }

    /**
     * Builder for ModelExporter.
     */
    public static class Builder {
        private NNModule model;
        private long[] inputShape;
        private Map<String, Object> metadata;

        private Builder() {
        }

        /**
         * Set the model to export.
         *
         * @param model model instance
         * @return this builder
         */
        public Builder model(NNModule model) {
            this.model = model;
            return this;
        }

        /**
         * Set input shape for tracing.
         *
         * @param shape input shape array
         * @return this builder
         */
        public Builder inputShape(long... shape) {
            this.inputShape = shape.clone();
            return this;
        }

        /**
         * Set metadata to include in exported model.
         *
         * @param metadata metadata map
         * @return this builder
         */
        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        /**
         * Build the ModelExporter instance.
         *
         * @return configured exporter
         */
        public ModelExporter build() {
            return new ModelExporter(this);
        }
    }
}
