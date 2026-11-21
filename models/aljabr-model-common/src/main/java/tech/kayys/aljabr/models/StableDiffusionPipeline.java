package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.models.diffusion.DDIMScheduler;
import tech.kayys.aljabr.models.diffusion.DiffusionSchedule;
import tech.kayys.aljabr.models.diffusion.DiffusionScheduler;
import tech.kayys.aljabr.models.diffusion.SchedulerStepParameters;

import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;
import java.util.function.Function;

import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.models.diffusion.DDIMScheduler;
import tech.kayys.aljabr.models.diffusion.DiffusionSchedule;
import tech.kayys.aljabr.models.diffusion.DiffusionScheduler;
import tech.kayys.aljabr.models.diffusion.SchedulerStepParameters;

/**
 * Text-to-image pipeline coordinating text encoding, denoising, and decoding.
 *
 * <p>The current CLIP and VAE classes in this module are compatibility
 * scaffolds rather than checkpoint-compatible Stable Diffusion components.
 * This pipeline therefore provides a reusable denoising orchestration surface,
 * not a claim of end-to-end Stable Diffusion checkpoint compatibility.</p>
 */
public class StableDiffusionPipeline extends NNModule {

    private static final int LATENT_CHANNELS = 4;
    private static final int LATENT_SIZE = 64;
    private static final int TIME_EMBEDDING_DIM = 320;
    private static final int TRAINING_TIMESTEPS = 1000;

    private final CLIP clip;
    private final UNet2DConditional unet;
    private final VAE vae;
    private final DiffusionScheduler scheduler;

    /** Retains the original vector-VAE constructor for existing callers. */
    public StableDiffusionPipeline(CLIP clip, UNet2DConditional unet, VAE vae) {
        this(clip, unet, vae, new DDIMScheduler(TRAINING_TIMESTEPS));
    }

    /**
     * Creates a pipeline with an injectable scheduler implementation.
     */
    public StableDiffusionPipeline(
            CLIP clip, UNet2DConditional unet, VAE vae, DiffusionScheduler scheduler) {
        this.clip = register("clip", Objects.requireNonNull(clip, "clip"));
        this.unet = register("unet", Objects.requireNonNull(unet, "unet"));
        this.vae = register("vae", Objects.requireNonNull(vae, "vae"));
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    }

    /**
     * Generates a sample using a fresh random seed and deterministic DDIM
     * updates ({@code eta = 0}).
     *
     * @param prompt text description
     * @param numSteps number of inference steps
     * @param guidance classifier-free guidance scale
     * @param progress optional callback receiving completed step count
     * @return decoded sample tensor
     */
    public Tensor generate(
            String prompt, int numSteps, float guidance, Consumer<Integer> progress) {
        return generate(prompt, numSteps, guidance,
                ThreadLocalRandom.current().nextLong(), 0.0f, progress);
    }

    /**
     * Generates a sample with reproducible initial and per-step noise.
     *
     * <p>For DDIM, {@code eta = 0} is deterministic for a fixed prompt,
     * model state, and seed. Positive eta enables stochastic DDIM updates.</p>
     *
     * @param prompt text description
     * @param numSteps number of inference steps
     * @param guidance classifier-free guidance scale
     * @param seed random seed
     * @param eta DDIM stochasticity; zero is deterministic
     * @param progress optional callback receiving completed step count
     * @return decoded sample tensor
     */
    public Tensor generate(
            String prompt,
            int numSteps,
            float guidance,
            long seed,
            float eta,
            Consumer<Integer> progress) {
        Objects.requireNonNull(prompt, "prompt");
        if (numSteps < 1) {
            throw new IllegalArgumentException("numSteps must be positive");
        }
        if (!Float.isFinite(guidance) || guidance < 0.0f) {
            throw new IllegalArgumentException("guidance must be finite and non-negative");
        }
        if (!Float.isFinite(eta) || eta < 0.0f) {
            throw new IllegalArgumentException("eta must be finite and non-negative");
        }

        DiffusionSchedule schedule = scheduler.createSchedule(numSteps);
        Tensor textEmbeds = clip.encode(prompt);
        Tensor unconditionalEmbeds = clip.encode("");
        Tensor latents = gaussianTensor(
                seed, 0, 1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE);

        try (textEmbeds; unconditionalEmbeds) {
            try {
                for (int stepIndex = 0; stepIndex < schedule.inferenceSteps(); stepIndex++) {
                    int timestep = schedule.timestepAt(stepIndex);
                    try (Tensor timestepEmbedding = timestepEmbedding(timestep);
                         Tensor noisePrediction = predictNoise(
                                 latents, timestepEmbedding, unconditionalEmbeds, textEmbeds, guidance)) {
                        SchedulerStepParameters parameters =
                                scheduler.stepParameters(schedule, stepIndex, eta);
                        Tensor stepNoise = parameters.sigma() == 0.0f
                                ? null
                                : gaussianTensor(seed, stepIndex + 1,
                                        1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE);
                        Tensor nextLatents;
                        if (stepNoise == null) {
                            nextLatents = schedulerStep(latents, noisePrediction, parameters, null);
                        } else {
                            try (stepNoise) {
                                nextLatents = schedulerStep(
                                        latents, noisePrediction, parameters, stepNoise);
                            }
                        }
                        latents.close();
                        latents = nextLatents;
                    }

                    if (progress != null) {
                        progress.accept(stepIndex + 1);
                    }
                }
                try (Tensor decoded = vae.decode(latents)) {
                    return decoded.squeeze();
                }
            } finally {
                latents.close();
            }
        }
    }

    private Tensor predictNoise(
            Tensor latents,
            Tensor timestep,
            Tensor unconditionalEmbeds,
            Tensor textEmbeds,
            float guidance) {
        try (Tensor unconditional = unet.forward(latents, timestep, unconditionalEmbeds);
             Tensor conditional = unet.forward(latents, timestep, textEmbeds);
             Tensor difference = conditional.sub(unconditional);
             Tensor guidedDifference = difference.mul(guidance)) {
            return unconditional.add(guidedDifference);
        }
    }

    private Tensor timestepEmbedding(int timestep) {
        float[] embedding = new float[TIME_EMBEDDING_DIM];
        int halfDim = TIME_EMBEDDING_DIM / 2;
        double logScale = Math.log(10000.0) / (halfDim - 1);
        for (int i = 0; i < halfDim; i++) {
            double frequency = Math.exp(-logScale * i);
            embedding[i] = (float) Math.sin(timestep * frequency);
            embedding[i + halfDim] = (float) Math.cos(timestep * frequency);
        }
        return Tensor.of(embedding, 1, TIME_EMBEDDING_DIM);
    }

    private Tensor schedulerStep(
            Tensor sample,
            Tensor epsilon,
            SchedulerStepParameters parameters,
            Tensor randomNoise) {
        float alpha = parameters.alphaCumprod();
        float previousAlpha = parameters.previousAlphaCumprod();

        try (Tensor scaledNoise = epsilon.mul((float) Math.sqrt(1.0f - alpha));
             Tensor numerator = sample.sub(scaledNoise);
             Tensor predictedOriginal = numerator.div((float) Math.sqrt(alpha));
             Tensor scaledOriginal = predictedOriginal.mul((float) Math.sqrt(previousAlpha));
             Tensor direction = epsilon.mul(parameters.directionCoefficient())) {
            Tensor deterministicSample = scaledOriginal.add(direction);
            if (parameters.sigma() == 0.0f) {
                return deterministicSample;
            }
            try (deterministicSample;
                 Tensor scaledRandomNoise = Objects.requireNonNull(
                         randomNoise, "randomNoise is required when eta produces non-zero sigma")
                         .mul(parameters.sigma())) {
                return deterministicSample.add(scaledRandomNoise);
            }
        }
    }

    private Tensor gaussianTensor(long seed, int stream, long... shape) {
        long elements = 1;
        for (long dimension : shape) {
            elements = Math.multiplyExact(elements, dimension);
        }
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Requested latent tensor is too large");
        }
        Random random = new Random(seed ^ (0x9E3779B97F4A7C15L * (stream + 1L)));
        float[] values = new float[(int) elements];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) random.nextGaussian();
        }
        return Tensor.of(values, shape);
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
                "Use generate(prompt, steps, guidance, ...) for diffusion inference");
    }
}
