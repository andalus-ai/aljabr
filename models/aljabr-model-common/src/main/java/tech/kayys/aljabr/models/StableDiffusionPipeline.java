package tech.kayys.aljabr.models;

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
 * Text-to-image diffusion orchestration over a text encoder, denoiser,
 * scheduler, and latent decoder.
 *
 * <p>The legacy {@link VAE} constructor remains available. New spatial
 * diffusion pipelines can use {@link AutoencoderKL}; its latent scaling
 * contract is applied by the decoder itself. The compact model components
 * remain architectural foundations and are not yet checkpoint-compatible
 * with Stable Diffusion, SDXL, SD3, or FLUX.</p>
 */
public class StableDiffusionPipeline extends NNModule {

    private static final int DEFAULT_LATENT_CHANNELS = 4;
    private static final int DEFAULT_LATENT_SIZE = 64;
    private static final int TIME_EMBEDDING_DIM = 320;
    private static final int TRAINING_TIMESTEPS = 1000;

    private final CLIP clip;
    private final UNet2DConditional unet;
    private final NNModule decoder;
    private final Function<Tensor, Tensor> decode;
    private final DiffusionScheduler scheduler;
    private final int latentChannels;

    /** Retains the original vector-VAE constructor for existing callers. */
    public StableDiffusionPipeline(CLIP clip, UNet2DConditional unet, VAE vae) {
        this(clip, unet, vae, vae::decode, DEFAULT_LATENT_CHANNELS,
                new DDIMScheduler(TRAINING_TIMESTEPS));
    }

    /** Creates a pipeline using the spatial AutoencoderKL decoder. */
    public StableDiffusionPipeline(
            CLIP clip, UNet2DConditional unet, AutoencoderKL autoencoder) {
        this(clip, unet, autoencoder, autoencoder::decode,
                autoencoder.config().latentChannels(), new DDIMScheduler(TRAINING_TIMESTEPS));
    }

    /** Legacy VAE constructor with an injectable scheduler. */
    public StableDiffusionPipeline(
            CLIP clip, UNet2DConditional unet, VAE vae, DiffusionScheduler scheduler) {
        this(clip, unet, vae, vae::decode, DEFAULT_LATENT_CHANNELS, scheduler);
    }

    /** Spatial AutoencoderKL constructor with an injectable scheduler. */
    public StableDiffusionPipeline(
            CLIP clip, UNet2DConditional unet,
            AutoencoderKL autoencoder, DiffusionScheduler scheduler) {
        this(clip, unet, autoencoder, autoencoder::decode,
                autoencoder.config().latentChannels(), scheduler);
    }

    private StableDiffusionPipeline(
            CLIP clip,
            UNet2DConditional unet,
            NNModule decoder,
            Function<Tensor, Tensor> decode,
            int latentChannels,
            DiffusionScheduler scheduler) {
        this.clip = register("clip", Objects.requireNonNull(clip, "clip"));
        this.unet = register("unet", Objects.requireNonNull(unet, "unet"));
        this.decoder = register("decoder", Objects.requireNonNull(decoder, "decoder"));
        this.decode = Objects.requireNonNull(decode, "decode");
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        if (latentChannels <= 0) {
            throw new IllegalArgumentException("latentChannels must be positive");
        }
        this.latentChannels = latentChannels;
        validateComponentBackends();
    }

    /**
     * Generates a sample using a fresh random seed and deterministic DDIM
     * updates ({@code eta = 0}).
     */
    public Tensor generate(
            String prompt, int numSteps, float guidance, Consumer<Integer> progress) {
        return generate(prompt, numSteps, guidance,
                ThreadLocalRandom.current().nextLong(), 0.0f, progress);
    }

    /**
     * Generates a sample with reproducible initial and per-step noise.
     *
     * @param prompt text description
     * @param numSteps number of inference steps
     * @param guidance classifier-free guidance scale
     * @param seed random seed
     * @param eta DDIM stochasticity; zero is deterministic
     * @param progress optional callback receiving completed step count
     * @return decoded image tensor
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
        Tensor textEmbeds = null;
        Tensor unconditionalEmbeds = null;
        Tensor latents = null;
        try {
            textEmbeds = clip.encode(prompt);
            unconditionalEmbeds = clip.encode("");
            latents = gaussianTensor(seed, 0,
                    1, latentChannels, DEFAULT_LATENT_SIZE, DEFAULT_LATENT_SIZE);

            for (int stepIndex = 0; stepIndex < schedule.inferenceSteps(); stepIndex++) {
                int timestep = schedule.timestepAt(stepIndex);
                try (Tensor timestepEmbedding = timestepEmbedding(timestep);
                     Tensor noisePrediction = predictNoise(
                             latents, timestepEmbedding, unconditionalEmbeds, textEmbeds, guidance)) {
                    SchedulerStepParameters parameters =
                            scheduler.stepParameters(schedule, stepIndex, eta);
                    Tensor nextLatents;
                    if (parameters.sigma() == 0.0f) {
                        nextLatents = schedulerStep(latents, noisePrediction, parameters, null);
                    } else {
                        try (Tensor stepNoise = gaussianTensor(seed, stepIndex + 1,
                                1, latentChannels, DEFAULT_LATENT_SIZE, DEFAULT_LATENT_SIZE)) {
                            nextLatents = schedulerStep(
                                    latents, noisePrediction, parameters, stepNoise);
                        }
                    }
                    latents.release();
                    latents = nextLatents;
                }

                if (progress != null) {
                    progress.accept(stepIndex + 1);
                }
            }
            return decode.apply(latents).squeeze();
        } finally {
            release(latents);
            release(textEmbeds);
            release(unconditionalEmbeds);
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
        return tensorOf(embedding, 1, TIME_EMBEDDING_DIM);
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
                         randomNoise, "randomNoise is required for stochastic DDIM")
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
        return tensorOf(values, shape);
    }

    private Tensor tensorOf(float[] values, long... shape) {
        return unet.computeBackend() == null
                ? Tensor.of(values, shape)
                : Tensor.of(values, unet.computeBackend(), shape);
    }

    private void validateComponentBackends() {
        if (unet.computeBackend() != null && decoder.computeBackend() != null
                && unet.computeBackend() != decoder.computeBackend()) {
            throw new IllegalArgumentException("Denoiser and decoder must use the same compute backend");
        }
    }

    private static void release(Tensor tensor) {
        if (tensor != null) {
            tensor.release();
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
                "Use generate(prompt, steps, guidance, ...) for diffusion inference");
    }
}
