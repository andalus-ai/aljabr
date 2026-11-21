package tech.kayys.aljabr.models;

import java.util.Objects;
import java.util.Random;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Downsample2d;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.Upsample2d;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Spatial variational autoencoder with a diagonal Gaussian latent posterior.
 *
 * <p>This is a configurable AutoencoderKL foundation for image diffusion. It
 * uses NCHW tensors, learned spatial down/up-sampling and a moments projection.
 * Its compact topology is intentionally not advertised as checkpoint-compatible
 * with any particular Stable Diffusion or FLUX VAE configuration.</p>
 *
 * <p>The public latent methods apply the configured scaling factor at the
 * diffusion boundary: {@link #encodeLatents(Tensor, long, boolean)} scales
 * posterior latents, while {@link #decode(Tensor)} reverses that scaling.</p>
 */
public final class AutoencoderKL extends NNModule {

    private final Config config;
    private final Conv2d encoderInput;
    private final Downsample2d downsample1;
    private final Conv2d encoderMiddle;
    private final Downsample2d downsample2;
    private final Conv2d moments;
    private final Conv2d postQuant;
    private final Upsample2d upsample1;
    private final Conv2d decoderMiddle;
    private final Upsample2d upsample2;
    private final Conv2d decoderOutput;

    public AutoencoderKL(ComputeBackend backend) {
        this(backend, Config.defaults());
    }

    public AutoencoderKL(ComputeBackend backend, Config config) {
        super(Objects.requireNonNull(backend, "backend must not be null"));
        this.config = Objects.requireNonNull(config, "config must not be null");

        int base = config.baseChannels();
        int latent = config.latentChannels();
        this.encoderInput = register("encoder_input",
                new Conv2d(backend, config.inChannels(), base, 3, 1, 1));
        this.downsample1 = register("downsample_1", new Downsample2d(backend, base));
        this.encoderMiddle = register("encoder_middle",
                new Conv2d(backend, base, base * 2, 3, 1, 1));
        this.downsample2 = register("downsample_2", new Downsample2d(backend, base * 2));
        this.moments = register("moments",
                new Conv2d(backend, base * 2, latent * 2, 3, 1, 1));

        this.postQuant = register("post_quant",
                new Conv2d(backend, latent, base * 2, 3, 1, 1));
        this.upsample1 = register("upsample_1", new Upsample2d(backend, 2.0f));
        this.decoderMiddle = register("decoder_middle",
                new Conv2d(backend, base * 2, base, 3, 1, 1));
        this.upsample2 = register("upsample_2", new Upsample2d(backend, 2.0f));
        this.decoderOutput = register("decoder_output",
                new Conv2d(backend, base, config.outChannels(), 3, 1, 1));
    }

    public record Config(int inChannels, int outChannels, int latentChannels,
            int baseChannels, float scalingFactor) {
        public Config {
            if (inChannels <= 0 || outChannels <= 0 || latentChannels <= 0 || baseChannels <= 0) {
                throw new IllegalArgumentException("channel dimensions must be positive");
            }
            if (!Float.isFinite(scalingFactor) || scalingFactor <= 0.0f) {
                throw new IllegalArgumentException("scalingFactor must be finite and positive");
            }
        }

        public static Config defaults() {
            return new Config(3, 3, 4, 32, 0.18215f);
        }
    }

    /** Diagonal Gaussian posterior over spatial latents, with NCHW mean/log-variance. */
    public record LatentDistribution(Tensor mean, Tensor logVariance) {
        public LatentDistribution {
            Objects.requireNonNull(mean, "mean must not be null");
            Objects.requireNonNull(logVariance, "logVariance must not be null");
            if (!java.util.Arrays.equals(mean.shape().dims(), logVariance.shape().dims())) {
                throw new IllegalArgumentException("mean and logVariance shapes must match");
            }
        }

        /** Returns the deterministic posterior mode. The returned tensor is caller-owned. */
        public Tensor mode() {
            return mean.add(0.0f);
        }

        /** Reparameterized sample using a reproducible Gaussian stream. */
        public Tensor sample(long seed) {
            long[] shape = mean.shape().dims();
            long elements = 1;
            for (long dimension : shape) {
                elements = Math.multiplyExact(elements, dimension);
            }
            if (elements > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("latent tensor is too large to sample");
            }
            Random random = new Random(seed);
            float[] noise = new float[(int) elements];
            for (int i = 0; i < noise.length; i += 2) {
                double u1 = Math.max(1.0e-12, random.nextDouble());
                double u2 = random.nextDouble();
                double radius = Math.sqrt(-2.0 * Math.log(u1));
                double angle = 2.0 * Math.PI * u2;
                noise[i] = (float) (radius * Math.cos(angle));
                if (i + 1 < noise.length) {
                    noise[i + 1] = (float) (radius * Math.sin(angle));
                }
            }
            Tensor epsilon = Tensor.of(noise, mean.backend(), shape);
            Tensor standardDeviation = logVariance.mul(0.5f).exp();
            try {
                return mean.add(epsilon.mul(standardDeviation));
            } finally {
                epsilon.release();
                standardDeviation.release();
            }
        }
    }

    /**
     * Encodes an image batch into posterior moments.
     *
     * @param image NCHW image tensor with spatial dimensions divisible by four
     * @return posterior distribution; caller owns both returned tensors
     */
    public LatentDistribution encode(Tensor image) {
        validateImage(image);
        Tensor h1 = null;
        Tensor h2 = null;
        Tensor h3 = null;
        Tensor h4 = null;
        Tensor parameters = null;
        Tensor mean = null;
        Tensor logVariance = null;
        try {
            h1 = encoderInput.forward(image).silu();
            h2 = downsample1.forward(h1).silu();
            h3 = encoderMiddle.forward(h2).silu();
            h4 = downsample2.forward(h3).silu();
            parameters = moments.forward(h4);
            long[] shape = parameters.shape().dims();
            long[] partShape = {shape[0], config.latentChannels(), shape[2], shape[3]};
            long[] meanOffsets = {0, 0, 0, 0};
            long[] varianceOffsets = {0, config.latentChannels(), 0, 0};
            mean = parameters.slice(meanOffsets, partShape);
            logVariance = parameters.slice(varianceOffsets, partShape);
            LatentDistribution result = new LatentDistribution(mean, logVariance);
            mean = null;
            logVariance = null;
            return result;
        } finally {
            release(h1);
            release(h2);
            release(h3);
            release(h4);
            release(parameters);
            release(mean);
            release(logVariance);
        }
    }

    /**
     * Encodes an image and returns a diffusion-scaled latent tensor.
     * The posterior mode is used when {@code sample} is false.
     */
    public Tensor encodeLatents(Tensor image, long seed, boolean sample) {
        LatentDistribution posterior = encode(image);
        Tensor latent = null;
        try {
            latent = sample ? posterior.sample(seed) : posterior.mode();
            return latent.mul(config.scalingFactor());
        } finally {
            if (latent != null) {
                latent.release();
            }
            posterior.mean().release();
            posterior.logVariance().release();
        }
    }

    /**
     * Decodes diffusion-scaled NCHW latents into an image tensor.
     * Output values are unbounded model values; callers may apply the
     * image-range conversion appropriate to their pipeline.
     */
    public Tensor decode(Tensor diffusionLatents) {
        Objects.requireNonNull(diffusionLatents, "diffusionLatents must not be null");
        validateInputBackend(diffusionLatents);
        if (diffusionLatents.shape().rank() != 4
                || diffusionLatents.shape().dim(1) != config.latentChannels()) {
            throw new IllegalArgumentException("decode expects NCHW latents with "
                    + config.latentChannels() + " channels");
        }

        Tensor unscaled = diffusionLatents.div(config.scalingFactor());
        Tensor h1 = null;
        Tensor h2 = null;
        Tensor h3 = null;
        Tensor h4 = null;
        try {
            h1 = postQuant.forward(unscaled).silu();
            h2 = upsample1.forward(h1);
            h3 = decoderMiddle.forward(h2).silu();
            h4 = upsample2.forward(h3);
            return decoderOutput.forward(h4);
        } finally {
            unscaled.release();
            release(h1);
            release(h2);
            release(h3);
            release(h4);
        }
    }

    @Override
    public Tensor forward(Tensor image) {
        LatentDistribution posterior = encode(image);
        Tensor latent = null;
        try {
            latent = posterior.mode().mul(config.scalingFactor());
            return decode(latent);
        } finally {
            release(latent);
            posterior.mean().release();
            posterior.logVariance().release();
        }
    }

    public Config config() {
        return config;
    }

    private void validateImage(Tensor image) {
        Objects.requireNonNull(image, "image must not be null");
        validateInputBackend(image);
        if (image.shape().rank() != 4 || image.shape().dim(1) != config.inChannels()) {
            throw new IllegalArgumentException("encode expects NCHW input with "
                    + config.inChannels() + " channels");
        }
        if (image.shape().dim(2) < 4 || image.shape().dim(3) < 4
                || image.shape().dim(2) % 4 != 0 || image.shape().dim(3) % 4 != 0) {
            throw new IllegalArgumentException("image height and width must be positive multiples of four");
        }
    }

    private static void release(Tensor tensor) {
        if (tensor != null) {
            tensor.release();
        }
    }
}
