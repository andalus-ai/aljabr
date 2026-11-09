package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.Embedding;
import tech.kayys.aljabr.core.nn.LayerNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.PositionalEncoding;
import tech.kayys.aljabr.core.nn.TransformerBlock;

/**
 * GPT-2 style autoregressive language model.
 *
 * <p>Decoder-only Transformer with causal (masked) self-attention.
 * Based on <em>"Language Models are Unsupervised Multitask Learners"</em>
 * (Radford et al., 2019).
 */
public final class GPT2 {

    private GPT2() {}

    /** @deprecated Use {@link #gpt2Small(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule gpt2Small(int vocabSize) {
        return gpt2Small(null, vocabSize);
    }

    /** GPT-2 Small: 12 layers, dModel=768, 12 heads, dFF=3072. */
    public static NNModule gpt2Small(ComputeBackend backend, int vocabSize) {
        return new GPT2Model(backend, 768, 12, 12, 3072, 1024, vocabSize, 0.1f);
    }

    /** @deprecated Use {@link #gpt2Medium(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule gpt2Medium(int vocabSize) {
        return gpt2Medium(null, vocabSize);
    }

    /** GPT-2 Medium: 24 layers, dModel=1024, 16 heads, dFF=4096. */
    public static NNModule gpt2Medium(ComputeBackend backend, int vocabSize) {
        return new GPT2Model(backend, 1024, 16, 24, 4096, 1024, vocabSize, 0.1f);
    }

    static final class GPT2Model extends NNModule {

        private final Embedding tokenEmbed;
        private final PositionalEncoding posEnc;
        private final TransformerBlock[] blocks;
        private final LayerNorm finalNorm;
        private final Linear lmHead;

        GPT2Model(
                ComputeBackend backend,
                int dModel,
                int nHeads,
                int nLayers,
                int dFF,
                int maxSeqLen,
                int vocabSize,
                float dropout) {
            super(backend);
            this.tokenEmbed = register("token_embed", backend == null
                    ? new Embedding(vocabSize, dModel)
                    : new Embedding(backend, vocabSize, dModel));
            this.posEnc = register("pos_enc", new PositionalEncoding(backend, dModel, maxSeqLen));
            this.blocks = new TransformerBlock[nLayers];
            for (int i = 0; i < nLayers; i++) {
                blocks[i] = register("block_" + i,
                        new TransformerBlock(backend, dModel, nHeads, dFF, dropout));
            }
            this.finalNorm = register("norm", backend == null
                    ? new LayerNorm(dModel)
                    : new LayerNorm(backend, dModel));
            this.lmHead = register("lm_head", backend == null
                    ? new Linear(dModel, vocabSize, false)
                    : new Linear(backend, dModel, vocabSize, false));
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = posEnc.forward(input);
            for (TransformerBlock block : blocks) {
                output = block.forward(output);
            }
            output = finalNorm.forward(output);
            return lmHead.forward(output);
        }
    }
}
