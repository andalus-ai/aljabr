package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorOps;
import tech.kayys.aljabr.core.nn.Embedding;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.RMSNorm;
import tech.kayys.aljabr.core.nn.RotaryEmbedding;

/**
 * LLaMA-style decoder-only language model.
 *
 * <p>Uses RMSNorm, SwiGLU feed-forward layers, and rotary positional embeddings.
 * The current attention and rotary implementations retain their existing
 * placeholder limitations; this class establishes backend-safe construction.
 */
public final class LLaMA {

    private LLaMA() {}

    /** @deprecated Use {@link #llama7B(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule llama7B(int vocabSize) {
        return llama7B(null, vocabSize);
    }

    public static NNModule llama7B(ComputeBackend backend, int vocabSize) {
        return new LLaMAModel(backend, 4096, 32, 32, 11008, 2048, vocabSize, 0.0f);
    }

    /** @deprecated Use {@link #llamaTiny(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule llamaTiny(int vocabSize) {
        return llamaTiny(null, vocabSize);
    }

    public static NNModule llamaTiny(ComputeBackend backend, int vocabSize) {
        return new LLaMAModel(backend, 256, 4, 4, 1024, 512, vocabSize, 0.0f);
    }

    static final class LLaMAModel extends NNModule {

        private final Embedding tokenEmbed;
        private final LLaMABlock[] blocks;
        private final RMSNorm finalNorm;
        private final Linear lmHead;

        LLaMAModel(
                ComputeBackend backend,
                int dModel,
                int nHeads,
                int nLayers,
                int dFF,
                int maxSeqLen,
                int vocabSize,
                float dropout) {
            super(backend);
            this.tokenEmbed = register("embed", backend == null
                    ? new Embedding(vocabSize, dModel)
                    : new Embedding(backend, vocabSize, dModel));
            this.blocks = new LLaMABlock[nLayers];
            for (int i = 0; i < nLayers; i++) {
                blocks[i] = register("block_" + i,
                        new LLaMABlock(backend, dModel, nHeads, dFF, maxSeqLen));
            }
            this.finalNorm = register("norm", backend == null
                    ? new RMSNorm(dModel, 1e-6f)
                    : new RMSNorm(backend, dModel, 1e-6f));
            this.lmHead = register("lm_head", backend == null
                    ? new Linear(dModel, vocabSize, false)
                    : new Linear(backend, dModel, vocabSize, false));
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = input;
            for (LLaMABlock block : blocks) {
                output = block.forward(output);
            }
            return lmHead.forward(finalNorm.forward(output));
        }
    }

    /** One pre-norm decoder block with attention and SwiGLU feed-forward layers. */
    static final class LLaMABlock extends NNModule {

        private final RMSNorm norm1;
        private final RMSNorm norm2;
        private final Linear wQ;
        private final Linear wK;
        private final Linear wV;
        private final Linear wO;
        private final Linear wGate;
        private final Linear wUp;
        private final Linear wDown;
        private final RotaryEmbedding rope;
        private final int nHeads;
        private final int headDim;

        LLaMABlock(ComputeBackend backend, int dModel, int nHeads, int dFF, int maxSeqLen) {
            super(backend);
            if (dModel <= 0 || nHeads <= 0 || dModel % nHeads != 0) {
                throw new IllegalArgumentException("dModel must be positive and divisible by nHeads");
            }
            this.nHeads = nHeads;
            this.headDim = dModel / nHeads;
            this.norm1 = register("norm1", backend == null
                    ? new RMSNorm(dModel, 1e-6f) : new RMSNorm(backend, dModel, 1e-6f));
            this.norm2 = register("norm2", backend == null
                    ? new RMSNorm(dModel, 1e-6f) : new RMSNorm(backend, dModel, 1e-6f));
            this.wQ = register("wQ", linear(backend, dModel, dModel));
            this.wK = register("wK", linear(backend, dModel, dModel));
            this.wV = register("wV", linear(backend, dModel, dModel));
            this.wO = register("wO", linear(backend, dModel, dModel));
            this.wGate = register("wGate", linear(backend, dModel, dFF));
            this.wUp = register("wUp", linear(backend, dModel, dFF));
            this.wDown = register("wDown", linear(backend, dFF, dModel));
            this.rope = register("rope", new RotaryEmbedding(backend, headDim, maxSeqLen));
        }

        private static Linear linear(ComputeBackend backend, int in, int out) {
            return backend == null
                    ? new Linear(in, out, false)
                    : new Linear(backend, in, out, false);
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = input.add(attention(norm1.forward(input)));
            return output.add(swiglu(norm2.forward(output)));
        }

        private Tensor attention(Tensor input) {
            long[] shape = input.shape().dims();
            int batch = Math.toIntExact(shape[0]);
            int sequence = Math.toIntExact(shape[1]);
            int dimension = Math.toIntExact(shape[2]);
            float scale = (float) (1.0 / Math.sqrt(headDim));

            Tensor q = reshape4d(wQ.forward(input), batch, sequence, nHeads, headDim);
            Tensor k = reshape4d(wK.forward(input), batch, sequence, nHeads, headDim);
            Tensor v = reshape4d(wV.forward(input), batch, sequence, nHeads, headDim);

            q = rope.apply(q, q);
            k = rope.apply(k, k);

            Tensor scores = TensorOps.einsum("bhid,bhjd->bhij", q, k).mul(scale);
            scores = applyCausalMask(scores, sequence);
            Tensor attention = scores.softmax();
            Tensor output = TensorOps.einsum("bhij,bhjd->bhid", attention, v);
            return wO.forward(output.reshape(batch, sequence, dimension));
        }

        private Tensor swiglu(Tensor input) {
            Tensor gate = wGate.forward(input).silu();
            Tensor up = wUp.forward(input);
            return wDown.forward(gate.mul(up));
        }

        private Tensor applyCausalMask(Tensor scores, int sequence) {
            float[] values = scores.toFloatArray().clone();
            long[] shape = scores.shape().dims();
            int batch = Math.toIntExact(shape[0]);
            int heads = Math.toIntExact(shape[1]);
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < heads; h++) {
                    for (int i = 0; i < sequence; i++) {
                        for (int j = i + 1; j < sequence; j++) {
                            values[b * heads * sequence * sequence
                                    + h * sequence * sequence + i * sequence + j] = Float.NEGATIVE_INFINITY;
                        }
                    }
                }
            }
            return tensorOf(values, shape);
        }

        private Tensor reshape4d(Tensor input, int batch, int sequence, int heads, int dimension) {
            float[] source = input.toFloatArray();
            float[] destination = new float[Math.multiplyExact(
                    Math.multiplyExact(Math.multiplyExact(batch, heads), sequence), dimension)];
            for (int b = 0; b < batch; b++) {
                for (int t = 0; t < sequence; t++) {
                    for (int h = 0; h < heads; h++) {
                        for (int d = 0; d < dimension; d++) {
                            destination[b * heads * sequence * dimension + h * sequence * dimension
                                    + t * dimension + d] =
                                    source[b * sequence * heads * dimension + t * heads * dimension
                                            + h * dimension + d];
                        }
                    }
                }
            }
            return tensorOf(destination, batch, heads, sequence, dimension);
        }
    }
}
