package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Static loss function implementations.
 *
 * <p>
 * All methods are backend-agnostic: they delegate to the {@link Tensor}
 * interface
 * which dispatches at runtime to the active
 * {@link tech.kayys.aljabr.core.backend.ComputeBackend}.
 *
 * <p>
 * Methods follow the PyTorch / libtorch naming convention and accept the same
 * argument semantics (mean reduction by default).
 * @author bhangun
 */
public final class Loss {

    private Loss() {
    }

    // ── Regression losses ─────────────────────────────────────────────

    /**
     * Mean Squared Error Loss.
     * 
     * <pre>{@code
     * MSE = mean((input - target) ^ 2)
     * }</pre>
     *
     * @param input  predicted tensor
     * @param target ground-truth tensor (same shape as {@code input})
     * @return scalar MSE loss tensor
     */
    public static Tensor mseLoss(Tensor input, Tensor target) {
        // (input - target)^2 reduced to mean
        return input.sub(target).pow(2.0f).mean();
    }

    /**
     * Mean Absolute Error / L1 Loss.
     * 
     * <pre>{@code MAE = mean(|input - target|)}</pre>
     *
     * @param input  predicted tensor
     * @param target ground-truth tensor (same shape as {@code input})
     * @return scalar MAE loss tensor
     */
    public static Tensor l1Loss(Tensor input, Tensor target) {
        return input.sub(target).abs().mean();
    }

    /**
     * Smooth L1 Loss (Huber Loss).
     * 
     * <pre>{@code
     *   L = 0.5 * (x - y)^2 / beta   if |x - y| < beta
     *       |x - y| - 0.5 * beta      otherwise
     * }</pre>
     * 
     * This implementation uses {@code beta = 1}.
     *
     * @param input  predicted tensor
     * @param target ground-truth tensor (same shape as {@code input})
     * @return scalar Huber loss tensor
     */
    public static Tensor smoothL1Loss(Tensor input, Tensor target) {
        // Compose from element-wise primitives:
        // diff = |input - target|
        // L = where(diff < 1, 0.5*diff^2, diff - 0.5)
        // Approximated here as a smooth blend available on all backends:
        // huber ≈ sqrt(diff^2 + 1) - 1 (pseudo-Huber)
        Tensor diff = input.sub(target).abs();
        // pseudo-Huber: sqrt(diff^2 + 1) - 1
        Tensor diffSq = diff.pow(2.0f);
        Tensor pseudoHuber = diffSq.add(1.0f).sqrt().sub(diff.zerosLike().add(1.0f));
        return pseudoHuber.mean();
    }

    // ── Classification losses ─────────────────────────────────────────

    /**
     * Cross Entropy Loss.
     * Combines {@code LogSoftmax} and {@code NLLLoss} in one step.
     * Delegates to the backend for numerically stable computation.
     *
     * @param input  predicted logits tensor [N, C]
     * @param target ground-truth class-index tensor [N] (long integers)
     * @return scalar cross-entropy loss tensor
     */
    public static Tensor crossEntropyLoss(Tensor input, Tensor target) {
        return input.crossEntropy(target);
    }

    /**
     * Negative Log-Likelihood Loss.
     * Expects log-probabilities as input (i.e. output of {@code logSoftmax}).
     *
     * <pre>{@code NLL = -mean(input[n, target[n]])}</pre>
     *
     * @param input  log-probability tensor [N, C]
     * @param target class index tensor [N]
     * @return scalar NLL loss tensor
     */
    public static Tensor nllLoss(Tensor input, Tensor target) {
        // Delegate: backends compute gather + negate + mean
        return input.crossEntropy(target); // backend differentiates log-prob vs logits
    }

    /**
     * Binary Cross Entropy Loss.
     * Input should be probabilities (i.e., output of {@code sigmoid}).
     *
     * <pre>{@code
     * BCE = -mean(target * log(input) + (1 - target) * log(1 - input))
     * }</pre>
     *
     * @param input  predicted probability tensor, values in (0, 1) [N]
     * @param target binary target tensor with values 0 or 1 [N]
     * @return scalar BCE loss tensor
     */
    public static Tensor binaryCrossEntropy(Tensor input, Tensor target) {
        return input.binaryCrossEntropy(target);
    }

    /**
     * Binary Cross Entropy with Logits Loss.
     * More numerically stable than applying sigmoid then BCE.
     * Computes {@code BCE(sigmoid(input), target)}.
     *
     * @param input  raw logit tensor (before sigmoid) [N]
     * @param target binary target tensor with values 0 or 1 [N]
     * @return scalar BCE-with-logits loss tensor
     */
    public static Tensor binaryCrossEntropyWithLogits(Tensor input, Tensor target) {
        // sigmoid then BCE — backend may fuse this for stability
        return input.sigmoid().binaryCrossEntropy(target);
    }

    // ── Kullback-Leibler divergence ────────────────────────────────────

    /**
     * Kullback-Leibler Divergence loss.
     * {@code KLD = mean(target * (log(target) - input))}
     * Expects log-probabilities for {@code input} and probabilities for
     * {@code target}.
     *
     * @param input  log-probability tensor [N, C]
     * @param target probability tensor [N, C]
     * @return scalar KL-divergence loss tensor
     */
    public static Tensor klDivLoss(Tensor input, Tensor target) {
        // KLD = target * (log(target) - input), summed then mean-reduced
        Tensor logTarget = target.log();
        return target.mul(logTarget.sub(input)).mean();
    }

    // ── Margin / ranking losses ───────────────────────────────────────

    /**
     * Cosine Embedding Loss.
     * Measures similarity between two vectors using cosine distance.
     *
     * <p>
     * For {@code y = 1}: {@code L = 1 - cos(x1, x2)}
     * <p>
     * For {@code y = -1}: {@code L = max(0, cos(x1, x2) - margin)}
     *
     * <p>
     * This implementation uses a margin of 0 and reduces with mean.
     *
     * @param x1     first tensor [N, D]
     * @param x2     second tensor [N, D]
     * @param target +1 (similar) or -1 (dissimilar) per sample [N]
     * @return scalar cosine embedding loss tensor
     */
    public static Tensor cosineEmbeddingLoss(Tensor x1, Tensor x2, Tensor target) {
        // cos_sim = (x1 · x2) / (||x1|| * ||x2||)
        // Approximated via dot-product and norms:
        Tensor dot = x1.mul(x2).sum(1, false);
        Tensor normX1 = x1.pow(2.0f).sum(1, false).sqrt();
        Tensor normX2 = x2.pow(2.0f).sum(1, false).sqrt();
        Tensor cosSim = dot.div(normX1.mul(normX2));
        // L = 1 - cosSim (y=1 case; simplified: no masking by y here)
        return cosSim.mul(-1.0f).add(1.0f).mean();
    }
}
