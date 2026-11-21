package tech.kayys.aljabr.models;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.SiLU;
import tech.kayys.aljabr.core.nn.Sequential;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Denoising Diffusion Probabilistic Model (DDPM) — learns to reverse a
 * Gaussian noise process to generate samples.
 *
 * <p>Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).
 * This implementation is a vector-data baseline; image latent diffusion uses
 * its own spatial denoiser and pipeline contracts.</p>
 */
public final class DDPM extends NNModule {

    private final int inputDim;
    private final int timesteps;
    private final float[] alphas;
    private final float[] alphasCumprod;
    private final float[] betas;

    /** Denoiser predicts epsilon from [x_t, normalized timestep]. */
    private final NNModule denoiser;

    /**
     * Creates a DDPM with a linear beta schedule on an explicitly selected backend.
     *
     * @param backend runtime-selected backend that owns all model and intermediate tensors
     * @param inputDim data dimension (e.g. 784 for MNIST)
     * @param hiddenDim denoiser hidden dimension
     * @param timesteps number of diffusion steps; must be at least two
     */
    public DDPM(ComputeBackend backend, int inputDim, int hiddenDim, int timesteps) {
        super(Objects.requireNonNull(backend, "backend must not be null"));
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive");
        }
        if (hiddenDim <= 0) {
            throw new IllegalArgumentException("hiddenDim must be positive");
        }
        if (timesteps < 2) {
            throw new IllegalArgumentException("timesteps must be at least 2");
        }

        this.inputDim = inputDim;
        this.timesteps = timesteps;
        this.betas = linearSchedule(1e-4f, 0.02f, timesteps);
        this.alphas = new float[timesteps];
        this.alphasCumprod = new float[timesteps];

        float cumulativeAlpha = 1f;
        for (int t = 0; t < timesteps; t++) {
            alphas[t] = 1f - betas[t];
            cumulativeAlpha *= alphas[t];
            alphasCumprod[t] = cumulativeAlpha;
        }

        this.denoiser = register("denoiser", new Sequential(backend,
                new Linear(backend, inputDim + 1, hiddenDim), new SiLU(backend),
                new Linear(backend, hiddenDim, hiddenDim), new SiLU(backend),
                new Linear(backend, hiddenDim, inputDim)));
    }

    /**
     * Computes the simplified DDPM training loss: MSE between predicted and
     * sampled noise. The input must be owned by this model's backend.
     *
     * @param x0 clean data samples with shape [batch, inputDim]
     * @return scalar MSE loss
     */
    public Tensor trainingLoss(Tensor x0) {
        Objects.requireNonNull(x0, "x0 must not be null");
        validateInputBackend(x0);
        if (x0.shape().rank() != 2 || x0.shape().dim(0) <= 0
                || x0.shape().dim(1) != inputDim) {
            throw new IllegalArgumentException(
                    "x0 must have shape [batch, " + inputDim + "]");
        }

        int batchSize = Math.toIntExact(x0.shape().dim(0));
        int[] sampledTimesteps = new int[batchSize];
        Random rng = new Random();
        for (int i = 0; i < batchSize; i++) {
            sampledTimesteps[i] = rng.nextInt(timesteps);
        }

        try (Tensor noise = TensorFactory.randn(computeBackend(), batchSize, inputDim)) {
            float[] clean = x0.toFloatArray();
            float[] noiseValues = noise.toFloatArray();
            float[] noisy = new float[batchSize * inputDim];

            for (int n = 0; n < batchSize; n++) {
                int timestep = sampledTimesteps[n];
                float signalScale = (float) Math.sqrt(alphasCumprod[timestep]);
                float noiseScale = (float) Math.sqrt(1f - alphasCumprod[timestep]);
                for (int d = 0; d < inputDim; d++) {
                    int index = n * inputDim + d;
                    noisy[index] = signalScale * clean[index] + noiseScale * noiseValues[index];
                }
            }

            try (Tensor modelInput = appendTimestep(noisy, sampledTimesteps, batchSize);
                 Tensor predictedNoise = denoiser.forward(modelInput);
                 Tensor difference = predictedNoise.sub(noise);
                 Tensor squaredError = difference.pow(2f)) {
                return squaredError.mean();
            }
        }
    }

    /** Samples a batch using a fresh, non-deterministic seed. */
    public Tensor sample(int batchSize) {
        return sample(batchSize, new Random().nextLong());
    }

    /**
     * Samples a batch reproducibly for a fixed model state and seed.
     *
     * <p>The same seed produces the same initial Gaussian noise and reverse
     * process noise. Reproducibility across different compute backends is not
     * guaranteed because floating-point kernels may differ.</p>
     */
    public Tensor sample(int batchSize, long seed) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }

        Random rng = new Random(seed);
        float[] sample = new float[batchSize * inputDim];
        for (int i = 0; i < sample.length; i++) {
            sample[i] = (float) rng.nextGaussian();
        }

        for (int t = timesteps - 1; t >= 0; t--) {
            int[] timestepValues = new int[batchSize];
            Arrays.fill(timestepValues, t);
            float[] predictedNoise;
            try (Tensor modelInput = appendTimestep(sample, timestepValues, batchSize);
                 Tensor prediction = denoiser.forward(modelInput)) {
                predictedNoise = prediction.toFloatArray();
            }

            float alpha = alphas[t];
            float beta = betas[t];
            float alphaBar = alphasCumprod[t];
            float previousAlphaBar = t == 0 ? 1f : alphasCumprod[t - 1];

            float meanScale = 1f / (float) Math.sqrt(alpha);
            float noiseScale = beta / (float) Math.sqrt(1f - alphaBar);
            float posteriorVariance = beta * (1f - previousAlphaBar) / (1f - alphaBar);
            float posteriorNoiseScale = (float) Math.sqrt(Math.max(0f, posteriorVariance));

            for (int i = 0; i < sample.length; i++) {
                float posteriorMean = meanScale * (sample[i] - noiseScale * predictedNoise[i]);
                sample[i] = posteriorMean;
                if (t > 0 && posteriorNoiseScale > 0f) {
                    sample[i] += posteriorNoiseScale * (float) rng.nextGaussian();
                }
            }
        }

        return TensorFactory.of(sample, computeBackend(), batchSize, inputDim);
    }

    @Override
    public Tensor forward(Tensor x) {
        return trainingLoss(x);
    }

    private Tensor appendTimestep(float[] values, int[] timestepValues, int batchSize) {
        float[] input = new float[batchSize * (inputDim + 1)];
        for (int n = 0; n < batchSize; n++) {
            System.arraycopy(values, n * inputDim, input, n * (inputDim + 1), inputDim);
            input[n * (inputDim + 1) + inputDim] = timestepValues[n] / (float) timesteps;
        }
        return TensorFactory.of(input, computeBackend(), batchSize, inputDim + 1);
    }

    private static float[] linearSchedule(float start, float end, int count) {
        float[] schedule = new float[count];
        for (int t = 0; t < count; t++) {
            schedule[t] = start + (end - start) * t / (count - 1);
        }
        return schedule;
    }
}
