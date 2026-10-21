package tech.kayys.aljabr.gguf.model.aljabr;

import tech.kayys.aljabr.core.nn.Embedding;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.RMSNorm;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import java.lang.foreign.ValueLayout;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.WeightAdapter;
import tech.kayys.aljabr.gguf.model.ModelConfig;

import java.util.ArrayList;
import java.util.List;
/**
 * 
 * Core class for aljabr module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class LlamaModel extends NNModule {
    private final Embedding embedTokens;
    private final List<LlamaDecoderLayer> layers = new ArrayList<>();
    private final RMSNorm norm;
    private final Linear lmHead;

    public LlamaModel(ModelConfig cfg, WeightAdapter weights) {
        // token_embd.weight
        embedTokens = new Embedding(weights.getWeight("token_embd.weight"));
        registerModule("embed_tokens", embedTokens);

        // Layers
        for (int i = 0; i < cfg.nLayers(); i++) {
            LlamaDecoderLayer layer = new LlamaDecoderLayer(cfg, weights, i);
            layers.add(layer);
            registerModule("layer_" + i, layer);
        }

        // output_norm.weight
        norm = new RMSNorm(weights.getWeight("output_norm.weight"), cfg.rmsNormEps());
        registerModule("norm", norm);

        // output.weight (or fallback to token_embd if tied)
        Tensor wOutput = weights.getWeight("output.weight");
        if (wOutput == null) {
            wOutput = weights.getWeight("token_embd.weight");
        }
        lmHead = new Linear(wOutput, null);
        registerModule("lm_head", lmHead);
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        return forward(inputIds, null);
    }

    public Tensor forward(Tensor inputIds, Tensor visionEmbeds) {
        // 1. Embed tokens
        Tensor hiddenStates = embedTokens.forward(inputIds);

        // Concatenate Vision Embeddings if present
        if (visionEmbeds != null) {
            hiddenStates = concatCpu(visionEmbeds, hiddenStates);
        }

        // 2. Pass through decoder layers
        for (LlamaDecoderLayer layer : layers) {
            hiddenStates = layer.forward(hiddenStates);
        }

        // 3. Final norm
        hiddenStates = norm.forward(hiddenStates);

        // 4. LM Head (only needed for the last token in generation, but we compute for
        // all here)
        Tensor logits = lmHead.forward(hiddenStates);

        return logits;
    }

    private Tensor concatCpu(Tensor a, Tensor b) {
        if (a.shape().dim(0) != b.shape().dim(0) || a.shape().dim(2) != b.shape().dim(2)) {
            throw new IllegalArgumentException("Batch or feature dimensions must match for concat");
        }

        // Make sure both are CPU tensors for extraction
        Tensor aCpu = a.device() != tech.kayys.aljabr.core.tensor.DeviceType.CPU
                ? a.backend().to(a, tech.kayys.aljabr.core.tensor.DeviceType.CPU)
                : a;
        Tensor bCpu = b.device() != tech.kayys.aljabr.core.tensor.DeviceType.CPU
                ? b.backend().to(b, tech.kayys.aljabr.core.tensor.DeviceType.CPU)
                : b;

        float[] arrA = ((DefaultTensor) aCpu).buffer().segment().toArray(ValueLayout.JAVA_FLOAT);
        float[] arrB = ((DefaultTensor) bCpu).buffer().segment().toArray(ValueLayout.JAVA_FLOAT);

        int batch = (int) a.shape().dim(0);
        int seqA = (int) a.shape().dim(1);
        int seqB = (int) b.shape().dim(1);
        int dim = (int) a.shape().dim(2);

        float[] arrOut = new float[batch * (seqA + seqB) * dim];

        for (int i = 0; i < batch; i++) {
            System.arraycopy(arrA, i * seqA * dim, arrOut, i * (seqA + seqB) * dim, seqA * dim);
            System.arraycopy(arrB, i * seqB * dim, arrOut, i * (seqA + seqB) * dim + seqA * dim, seqB * dim);
        }

        Tensor out = TensorFactory.of(arrOut, batch, seqA + seqB, dim);
        if (a.device() != tech.kayys.aljabr.core.tensor.DeviceType.CPU) {
            out = a.backend().to(out, a.device());
        }
        return out;
    }
}
