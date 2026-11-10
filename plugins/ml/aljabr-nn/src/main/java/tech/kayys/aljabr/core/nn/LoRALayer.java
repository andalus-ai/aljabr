package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Low-Rank Adaptation (LoRA) layer.
 */
public class LoRALayer extends NNModule {

    private final Linear baseLayer;
    private final int rank;
    private final double alpha;
    private final double scaling;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public LoRALayer(Linear baseLayer, int rank, double alpha) {
        this(requireBaseBackend(baseLayer), baseLayer, rank, alpha);
    }

    public LoRALayer(ComputeBackend backend, Linear baseLayer, int rank, double alpha) {
        super(Objects.requireNonNull(backend, "backend must not be null"));
        this.baseLayer = Objects.requireNonNull(baseLayer, "baseLayer must not be null");
        if (rank <= 0) {
            throw new IllegalArgumentException("rank must be positive");
        }
        Tensor baseWeight = Objects.requireNonNull(baseLayer.parameters().get("weight"),
                "base layer must contain a weight parameter");
        if (baseWeight.backend() != backend) {
            throw new IllegalArgumentException("Base Linear layer uses a different compute backend");
        }
        this.rank = rank;
        this.alpha = alpha;
        this.scaling = alpha / rank;

        long inFeatures = baseLayer.inFeatures();
        long outFeatures = baseLayer.outFeatures();
        registerParameter("lora_a", randomNormal(inFeatures, rank).mul((float) Math.sqrt(2.0 / inFeatures)));
        registerParameter("lora_b", zeros(rank, outFeatures));

        for (Tensor param : baseLayer.getParameters()) {
            param.setRequiresGrad(false);
        }
        registerModule("base", baseLayer);
    }

    private static ComputeBackend requireBaseBackend(Linear baseLayer) {
        Linear required = Objects.requireNonNull(baseLayer, "baseLayer must not be null");
        Tensor weight = Objects.requireNonNull(required.parameters().get("weight"),
                "base layer must contain a weight parameter");
        return weight.backend();
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        Tensor baseOutput = baseLayer.forward(input);
        Tensor loraA = parameters.get("lora_a");
        Tensor loraB = parameters.get("lora_b");
        Tensor loraOutput = input.matmul(loraA).matmul(loraB).mul((float) scaling);
        return baseOutput.add(loraOutput);
    }

    public Linear getBaseLayer() { return baseLayer; }
    public int getRank() { return rank; }
    public double getAlpha() { return alpha; }
    public double getScaling() { return scaling; }

    @Override
    public String toString() {
        return String.format("LoRALayer(base=%s, rank=%d, alpha=%.1f)", baseLayer, rank, alpha);
    }
}
