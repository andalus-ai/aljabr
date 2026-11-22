package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Maps named Hugging Face EmbeddingGemma 2 audio-tower tensors to the Java
 * reference encoder's explicit weight bundle.
 *
 * <p>The mapper is storage-format agnostic: a Safetensors or other checkpoint
 * reader supplies decoded dense tensors. It performs no I/O and never
 * initializes missing trainable parameters. It validates every consumed
 * tensor's exact shape and rejects unknown tensors in the audio_tower
 * namespace so checkpoint drift cannot silently produce partial weights.</p>
 */
public final class EmbeddingGemma2AudioCheckpointMapper {
    private static final String PREFIX = "audio_tower.";

    private EmbeddingGemma2AudioCheckpointMapper() {
    }

    public static EmbeddingGemma2AudioEncoder.Weights map(
            EmbeddingGemma2AudioConfig config,
            Map<String, CheckpointTensor> tensors) {
        Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        if (config.subsamplingConvolutionChannels().size() != 2) {
            throw new IllegalArgumentException(
                    "Gemma4 audio checkpoint mapping requires exactly two subsampling convolution channel sizes");
        }
        Mapping mapping = new Mapping(tensors);

        int firstChannels = config.subsamplingConvolutionChannels().get(0);
        int secondChannels = config.subsamplingConvolutionChannels().get(1);
        int projectionFrequency = downsample(downsample(config.featureSize()));
        int projectionWidth = Math.multiplyExact(projectionFrequency, secondChannels);

        EmbeddingGemma2AudioSubsampling.Weights subsampling =
                new EmbeddingGemma2AudioSubsampling.Weights(
                        new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                                mapping.conv2d(PREFIX + "subsample_conv_projection.layer0.conv.weight",
                                        firstChannels, 1),
                                mapping.vector(PREFIX + "subsample_conv_projection.layer0.norm.weight",
                                        firstChannels)),
                        new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                                mapping.conv2d(PREFIX + "subsample_conv_projection.layer1.conv.weight",
                                        secondChannels, firstChannels),
                                mapping.vector(PREFIX + "subsample_conv_projection.layer1.norm.weight",
                                        secondChannels)),
                        mapping.matrix(PREFIX + "subsample_conv_projection.input_proj_linear.weight",
                                config.hiddenSize(), projectionWidth));

        List<EmbeddingGemma2AudioConformerLayer.Weights> layers =
                new ArrayList<>(config.numHiddenLayers());
        for (int index = 0; index < config.numHiddenLayers(); index++) {
            String layer = PREFIX + "layers." + index + ".";
            int hidden = config.hiddenSize();
            int intermediate = Math.multiplyExact(hidden, 4);
            int headSize = hidden / config.numAttentionHeads();

            EmbeddingGemma2AudioFeedForward.Weights feedForward1 =
                    feedForward(mapping, layer + "feed_forward1.", hidden, intermediate);
            EmbeddingGemma2AudioFeedForward.Weights feedForward2 =
                    feedForward(mapping, layer + "feed_forward2.", hidden, intermediate);

            EmbeddingGemma2AudioAttention.Weights attention =
                    new EmbeddingGemma2AudioAttention.Weights(
                            attentionLinear(mapping, layer + "self_attn.q_proj.", hidden, hidden),
                            attentionLinear(mapping, layer + "self_attn.k_proj.", hidden, hidden),
                            attentionLinear(mapping, layer + "self_attn.v_proj.", hidden, hidden),
                            attentionLinear(mapping, layer + "self_attn.post.", hidden, hidden),
                            mapping.matrix(layer + "self_attn.relative_k_proj.weight", hidden, hidden),
                            mapping.vector(layer + "self_attn.per_dim_scale", headSize));

            EmbeddingGemma2AudioLightConv1d.Weights lightConv =
                    new EmbeddingGemma2AudioLightConv1d.Weights(
                            lightConvLinear(mapping, layer + "lconv1d.linear_start.", hidden * 2, hidden),
                            lightConvLinear(mapping, layer + "lconv1d.linear_end.", hidden, hidden),
                            mapping.depthwiseKernel(layer + "lconv1d.depthwise_conv1d.weight",
                                    hidden, config.convolutionKernelSize()),
                            mapping.vector(layer + "lconv1d.pre_layer_norm.weight", hidden),
                            mapping.vector(layer + "lconv1d.conv_norm.weight", hidden));

            layers.add(new EmbeddingGemma2AudioConformerLayer.Weights(
                    feedForward1,
                    mapping.vector(layer + "norm_pre_attn.weight", hidden),
                    attention,
                    mapping.vector(layer + "norm_post_attn.weight", hidden),
                    lightConv,
                    feedForward2,
                    mapping.vector(layer + "norm_out.weight", hidden)));
        }

        float[][] outputProjection = mapping.matrix(
                PREFIX + "output_proj.weight", config.outputProjectionSize(), config.hiddenSize());
        float[] outputProjectionBias = mapping.vector(
                PREFIX + "output_proj.bias", config.outputProjectionSize());

        mapping.rejectUnknownAudioTensors();
        return new EmbeddingGemma2AudioEncoder.Weights(
                subsampling, layers, outputProjection, outputProjectionBias);
    }

    private static EmbeddingGemma2AudioFeedForward.Weights feedForward(
            Mapping mapping, String prefix, int hidden, int intermediate) {
        return new EmbeddingGemma2AudioFeedForward.Weights(
                feedForwardLinear(mapping, prefix + "ffw_layer_1.", intermediate, hidden),
                feedForwardLinear(mapping, prefix + "ffw_layer_2.", hidden, intermediate),
                mapping.vector(prefix + "pre_layer_norm.weight", hidden),
                mapping.vector(prefix + "post_layer_norm.weight", hidden));
    }

    private static EmbeddingGemma2AudioFeedForward.LinearWeights feedForwardLinear(
            Mapping mapping, String prefix, int rows, int columns) {
        return new EmbeddingGemma2AudioFeedForward.LinearWeights(
                mapping.matrix(prefix + "linear.weight", rows, columns),
                mapping.clipRange(prefix));
    }

    private static EmbeddingGemma2AudioLightConv1d.LinearWeights lightConvLinear(
            Mapping mapping, String prefix, int rows, int columns) {
        return new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                mapping.matrix(prefix + "linear.weight", rows, columns),
                mapping.lightConvClipRange(prefix));
    }

    private static EmbeddingGemma2AudioAttention.LinearWeights attentionLinear(
            Mapping mapping, String prefix, int rows, int columns) {
        return new EmbeddingGemma2AudioAttention.LinearWeights(
                mapping.matrix(prefix + "linear.weight", rows, columns),
                mapping.attentionClipRange(prefix));
    }

    private static int downsample(int size) {
        return (size + 1) / 2;
    }

    private static final class Mapping {
        private final Map<String, CheckpointTensor> tensors;
        private final Set<String> consumed = new HashSet<>();

        private Mapping(Map<String, CheckpointTensor> tensors) {
            this.tensors = tensors;
            for (Map.Entry<String, CheckpointTensor> entry : tensors.entrySet()) {
                if (entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null) {
                    throw new IllegalArgumentException(
                            "checkpoint tensor map must contain non-blank names and non-null tensors");
                }
            }
        }

        private CheckpointTensor required(String name) {
            CheckpointTensor tensor = tensors.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException("missing required audio checkpoint tensor: " + name);
            }
            consumed.add(name);
            return tensor;
        }

        private float[][] matrix(String name, int rows, int columns) {
            CheckpointTensor tensor = required(name);
            requireShape(name, tensor, rows, columns);
            float[] values = tensor.values();
            float[][] result = new float[rows][columns];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(values, row * columns, result[row], 0, columns);
            }
            return result;
        }

        private float[] vector(String name, int length) {
            CheckpointTensor tensor = required(name);
            requireShape(name, tensor, length);
            return tensor.values();
        }

        private float[][][][] conv2d(String name, int outChannels, int inChannels) {
            CheckpointTensor tensor = required(name);
            requireShape(name, tensor, outChannels, inChannels, 3, 3);
            float[] flat = tensor.values();
            float[][][][] result = new float[outChannels][inChannels][3][3];
            int offset = 0;
            for (int out = 0; out < outChannels; out++) {
                for (int in = 0; in < inChannels; in++) {
                    for (int y = 0; y < 3; y++) {
                        System.arraycopy(flat, offset, result[out][in][y], 0, 3);
                        offset += 3;
                    }
                }
            }
            return result;
        }

        private float[][] depthwiseKernel(String name, int channels, int kernelSize) {
            CheckpointTensor tensor = required(name);
            requireShape(name, tensor, channels, 1, kernelSize);
            float[] flat = tensor.values();
            float[][] result = new float[channels][kernelSize];
            for (int channel = 0; channel < channels; channel++) {
                System.arraycopy(flat, channel * kernelSize, result[channel], 0, kernelSize);
            }
            return result;
        }

        private EmbeddingGemma2AudioFeedForward.ClipRange clipRange(String prefix) {
            return new EmbeddingGemma2AudioFeedForward.ClipRange(
                    scalar(prefix + "input_min"), scalar(prefix + "input_max"),
                    scalar(prefix + "output_min"), scalar(prefix + "output_max"));
        }

        private EmbeddingGemma2AudioLightConv1d.ClipRange lightConvClipRange(String prefix) {
            return new EmbeddingGemma2AudioLightConv1d.ClipRange(
                    scalar(prefix + "input_min"), scalar(prefix + "input_max"),
                    scalar(prefix + "output_min"), scalar(prefix + "output_max"));
        }

        private EmbeddingGemma2AudioAttention.ClipRange attentionClipRange(String prefix) {
            return new EmbeddingGemma2AudioAttention.ClipRange(
                    scalar(prefix + "input_min"), scalar(prefix + "input_max"),
                    scalar(prefix + "output_min"), scalar(prefix + "output_max"));
        }

        private float scalar(String name) {
            CheckpointTensor tensor = required(name);
            int[] shape = tensor.shape();
            if (!(shape.length == 0 || (shape.length == 1 && shape[0] == 1))) {
                throw new IllegalArgumentException(name + " must be a scalar or [1] tensor");
            }
            float[] values = tensor.values();
            if (values.length != 1) throw new IllegalArgumentException(name + " must contain one value");
            return values[0];
        }

        private void rejectUnknownAudioTensors() {
            for (String name : tensors.keySet()) {
                if (name.startsWith(PREFIX) && !consumed.contains(name)) {
                    throw new IllegalArgumentException(
                            "unsupported or unmapped audio checkpoint tensor: " + name);
                }
            }
        }

        private static void requireShape(String name, CheckpointTensor tensor, int... expected) {
            int[] actual = tensor.shape();
            if (!java.util.Arrays.equals(actual, expected)) {
                throw new IllegalArgumentException(name + " expected shape "
                        + java.util.Arrays.toString(expected) + " but was "
                        + java.util.Arrays.toString(actual));
            }
        }
    }
}
