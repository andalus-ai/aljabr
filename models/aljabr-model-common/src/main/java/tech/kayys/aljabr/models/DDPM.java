package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.*;

import java.util.Objects;
import java.util.Random;

/**
 * Denoising Diffusion Probabilistic Model (DDPM) — learns to reverse a
 * Gaussian noise process to generate samples.
 *
 * <p>Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).
 * This implementation is a vector-data baseline; image latent diffusion uses
 * its own spatial denoiser and pipeline contracts.
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
     * Creates a DDPM with a linear beta schedule.
     *
     * @param inputDim data dimension (e.g. 784 for MNIST)
     * @param hiddenDim denoiser hidden dimension
     * @param timesteps number of diffusion steps; must be at least two
     */
    public DDPM(int inputDim, int hiddenDim, int timesteps) {
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

        this.denoiser = register("denoiser", new Sequential(
                new Linear(inputDim + 1, hiddenDim), new SiLU(),
                new Linear(hiddenDim, hiddenDim), new SiLU(),
                new Linear(hiddenDim, inputDim)));
    }

    /**
     * Computes the simplified DDPM training loss: MSE between predicted and
     * sampled noise.
     *
     * @param x0 clean data samples with shape [batch, inputDim]
     * @return scalar MSE loss
     */
    public Tensor trainingLoss(Tensor x0) {
        Objects.requireNonNull(x0, "x0 must not be null");
        if (x0.shape().rank() != 2 || x0.shape().dim(0) <= 0
                || x0.shape().dim(1) != inputDim) {
            throw new IllegalArgumentException(
                    "x0 must have shape [batch, " + inputDim + "]");
        }

        int batchSize = (int) x0.shape().dim(0);
        int dimension = inputDim;

        int[] sampledTimesteps = new int[batchSize];
        Random rng = new Random();
        for (int i = 0; i < batchSize; i++) {
            sampledTimesteps[i] = rng.nextInt(timesteps);
        }

        Tensor noise = Tensor.randn(batchSize, dimension);
        float[] clean = x0.toFloatArray();
        float[] noiseValues = noise.toFloatArray();
        float[] noisy = new float[batchSize * dimension];

        for (int n = 0; n < batchSize; n++) {
            int timestep = sampledTimesteps[n];
            float signalScale = (float) Math.sqrt(alphasCumprod[timestep]);
            float noiseScale = (float) Math.sqrt(1f - alphasCumprod[timestep]);
            for (int d = 0; d < dimension; d++) {
                int index = n * dimension + d;
                noisy[index] = signalScale * clean[index] + noiseScale * noiseValues[index];
            }
        }

        Tensor modelInput = appendTimestep(noisy, sampledTimesteps, batchSize);
        Tensor predictedNoise = denoiser.forward(modelInput);
        return predictedNoise.sub(noise).pow(2f).mean();
    }

    /**
     * Samples a batch using a fresh, non-deterministic seed.
     *
     * @param batchSize number of samples
     * @return generated samples with shape [batchSize, inputDim]
     */
    public Tensor sample(int batchSize) {
        return sample(batchSize, new Random().nextLong());
    }

    /**
     * Samples a batch reproducibly for a fixed model state and seed.
     *
     * <p>The same seed produces the same initial Gaussian noise and reverse
     * process noise. Reproducibility across different compute backends is not
     * guaranteed because floating-point kernels may differ.
     *
     * @param batchSize number of samples
     * @param seed random seed for all sampling noise
     * @return generated samples with shape [batchSize, inputDim]
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
            java.util.Arrays.fill(timestepValues, t);
            float[] predictedNoise;
            try (Tensor modelInput = appendTimestep(sample, timestepValues, batchSize);
                 Tensor prediction = denoiser.forward(modelInput)) {
                predictedNoise = prediction.toFloatArray();
            }

            float alpha = alphas[t];
            float beta = betas[t];
            float alphaBar = alphasCumprod[t];
            float previousAlphaBar = t == 0 ? 1f : alphasCumprod[t - 1];

            // Posterior mean:
            // mu_theta = 1/sqrt(alpha_t) *
            //           (x_t - beta_t/sqrt(1-alphaBar_t) * epsilon_theta)
            float meanScale = 1f / (float) Math.sqrt(alpha);
            float noiseScale = beta / (float) Math.sqrt(1f - alphaBar);

            // DDPM posterior variance beta_tilde_t. It is zero at t=0.
            float posteriorVariance = beta * (1f - previousAlphaBar)
                    / (1f - alphaBar);
            float posteriorNoiseScale = (float) Math.sqrt(Math.max(0f, posteriorVariance));

            for (int i = 0; i < sample.length; i++) {
                float posteriorMean = meanScale * (sample[i] - noiseScale * predictedNoise[i]);
                sample[i] = posteriorMean;
                if (t > 0 && posteriorNoiseScale > 0f) {
                    sample[i] += posteriorNoiseScale * (float) rng.nextGaussian();
                }
            }
        }

        return Tensor.of(sample, batchSize, inputDim);
    }

    @Override
    public Tensor forward(Tensor x) {
        return trainingLoss(x);
    }

    private Tensor appendTimestep(float[] values, int[] timestepValues, int batchSize) {
        float[] input = new float[batchSize * (inputDim + 1)];
        for (int n = 0; n < batchSize; n++) {
            System.arraycopy(values, n * inputDim, input, n * (inputDim + 1), inputDim);
            input[n * (inputDim + 1) + inputDim] =
                    timestepValues[n] / (float) timesteps;
        }
        return Tensor.of(input, batchSize, inputDim + 1);
    }

    private static float[] linearSchedule(float start, float end, int count) {
        float[] schedule = new float[count];
        for (int t = 0; t < count; t++) {
            schedule[t] = start + (end - start) * t / (count - 1);
        }
        return schedule;
    }
}
