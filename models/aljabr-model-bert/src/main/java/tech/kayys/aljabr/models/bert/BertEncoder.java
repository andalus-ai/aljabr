package tech.kayys.aljabr.models.bert;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.TransformerEncoderLayer;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Reusable bidirectional BERT encoder for inference.
 *
 * <p>Accepts Hugging Face-style token IDs and binary attention masks
 * ({@code 1} for a real token, {@code 0} for padding). The mask is converted
 * to a per-example additive attention mask so padded keys cannot contribute
 * to any query. The returned tensor contains the final hidden states in
 * {@code [batch, sequence, hiddenSize]} layout.</p>
 *
 * <p>Checkpoint weights must be loaded before meaningful inference. The
 * underlying encoder layer currently omits dropout and is intended for
 * deterministic inference.</p>
 */
public final class BertEncoder extends NNModule {
    private static final float MASKED_ATTENTION_BIAS = -1.0e9f;

    private final BertConfig config;
    private final BertInputEmbeddings inputEmbeddings;
    private final List<TransformerEncoderLayer> layers;

    public BertEncoder(ComputeBackend backend, BertConfig config) {
        super(Objects.requireNonNull(backend, "backend must not be null"));
        this.config = Objects.requireNonNull(config, "config must not be null");
        inputEmbeddings = registerModule("embeddings", new BertInputEmbeddings(backend, config));
        List<TransformerEncoderLayer> encoderLayers = new ArrayList<>(config.layerCount());
        for (int i = 0; i < config.layerCount(); i++) {
            encoderLayers.add(registerModule("encoder_layer_" + i,
                    new TransformerEncoderLayer(backend, config.hiddenSize(),
                            config.attentionHeadCount(), config.intermediateSize(), 0.0f,
                            (float) config.layerNormEpsilon())));
        }
        layers = List.copyOf(encoderLayers);
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        return forward(inputIds, null, null);
    }

    /**
     * @param inputIds I32 token IDs [batch, sequence]
     * @param attentionMask optional I32 binary mask [batch, sequence]
     * @param tokenTypeIds optional I32 segment IDs [batch, sequence]
     */
    public Tensor forward(Tensor inputIds, Tensor attentionMask, Tensor tokenTypeIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        validateBackend(inputIds, "inputIds");
        if (inputIds.dtype() != DType.I32 || inputIds.shape().rank() != 2) {
            throw new IllegalArgumentException("inputIds must be I32 with shape [batch, sequence]");
        }
        if (attentionMask != null) validateAlignedInput(inputIds, attentionMask, "attentionMask");
        if (tokenTypeIds != null) validateAlignedInput(inputIds, tokenTypeIds, "tokenTypeIds");

        Tensor additiveMask = null;
        Tensor hidden = null;
        try {
            if (attentionMask != null) {
                additiveMask = createAdditiveAttentionMask(attentionMask);
            }
            hidden = inputEmbeddings.forward(inputIds, tokenTypeIds);
            for (TransformerEncoderLayer layer : layers) {
                Tensor next = layer.forward(hidden, additiveMask);
                hidden.release();
                hidden = next;
            }
            Tensor result = hidden;
            hidden = null;
            return result;
        } finally {
            if (hidden != null) hidden.release();
            if (additiveMask != null) additiveMask.release();
        }
    }

    private Tensor createAdditiveAttentionMask(Tensor attentionMask) {
        if (attentionMask.dtype() != DType.I32) {
            throw new IllegalArgumentException("attentionMask must have dtype I32");
        }
        long batch = attentionMask.shape().dim(0);
        long sequence = attentionMask.shape().dim(1);
        int batchSize = Math.toIntExact(batch);
        int sequenceSize = Math.toIntExact(sequence);
        int[] values = attentionMask.toIntArray();
        float[] additive = new float[Math.toIntExact(Math.multiplyExact(
                Math.multiplyExact(batch, sequence), sequence))];
        for (int b = 0; b < batchSize; b++) {
            for (int key = 0; key < sequenceSize; key++) {
                int valid = values[b * sequenceSize + key];
                if (valid != 0 && valid != 1) {
                    throw new IllegalArgumentException("attentionMask values must be binary (0 or 1)");
                }
                if (valid == 0) {
                    int rowStart = b * sequenceSize * sequenceSize;
                    for (int query = 0; query < sequenceSize; query++) {
                        additive[rowStart + query * sequenceSize + key] = MASKED_ATTENTION_BIAS;
                    }
                }
            }
        }
        return TensorFactory.of(additive, attentionMask.backend(), batch, sequence, sequence);
    }

    private void validateAlignedInput(Tensor inputIds, Tensor candidate, String name) {
        validateBackend(candidate, name);
        if (candidate.dtype() != DType.I32
                || !java.util.Arrays.equals(candidate.shape().dims(), inputIds.shape().dims())) {
            throw new IllegalArgumentException(name + " must be I32 and match inputIds shape");
        }
    }

    private void validateBackend(Tensor tensor, String name) {
        if (tensor.backend() != computeBackend()) {
            throw new IllegalArgumentException(name + " must use the encoder's compute backend");
        }
    }

    /**
     * Loads a complete Hugging Face BERT encoder checkpoint after storage-specific
     * readers have converted tensors to dense F32 payloads. The checkpoint is
     * validated and copied into this module atomically by NNModule.loadStateDict.
     */
    public void loadCheckpoint(Map<String, BertCheckpointMapper.CheckpointTensor> checkpointTensors) {
        BertCheckpointMapper mapper = new BertCheckpointMapper();
        Map<String, BertCheckpointMapper.CheckpointTensor> mapped = mapper.map(config, checkpointTensors);
        Map<String, Tensor> state = new java.util.LinkedHashMap<>();
        try {
            mapped.forEach((name, tensor) -> {
                int[] shape = tensor.shape();
                long[] dimensions = new long[shape.length];
                for (int i = 0; i < shape.length; i++) dimensions[i] = shape[i];
                state.put(name, TensorFactory.of(tensor.values(), computeBackend(), dimensions));
            });
            loadStateDict(state, true);
        } finally {
            state.values().forEach(Tensor::release);
        }
    }

    public BertConfig config() {
        return config;
    }

    public int layerCount() {
        return layers.size();
    }
}
