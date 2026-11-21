package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class AutoencoderKLTest {

    @Test
    void encodesSpatialPosteriorAndDecodesDiffusionScaledLatents() {
        CpuBackend backend = new CpuBackend();
        AutoencoderKL model = new AutoencoderKL(backend,
                new AutoencoderKL.Config(3, 3, 4, 8, 0.18215f));
        Tensor image = TensorFactory.of(new float[3 * 8 * 8], backend, 1, 3, 8, 8);
        Tensor latent = null;
        Tensor decoded = null;
        try {
            AutoencoderKL.LatentDistribution posterior = model.encode(image);
            try {
                assertEquals(4, posterior.mean().shape().dim(1));
                assertEquals(2, posterior.mean().shape().dim(2));
                assertEquals(2, posterior.mean().shape().dim(3));
                assertArrayEquals(posterior.mean().shape().dims(),
                        posterior.logVariance().shape().dims());
                latent = posterior.sample(1234L).mul(model.config().scalingFactor());
            } finally {
                posterior.mean().release();
                posterior.logVariance().release();
            }

            decoded = model.decode(latent);
            assertEquals(1, decoded.shape().dim(0));
            assertEquals(3, decoded.shape().dim(1));
            assertEquals(8, decoded.shape().dim(2));
            assertEquals(8, decoded.shape().dim(3));
        } finally {
            if (decoded != null) decoded.release();
            if (latent != null) latent.release();
            image.release();
            model.close();
        }
    }

    @Test
    void posteriorSamplingIsReproducibleForTheSameSeed() {
        CpuBackend backend = new CpuBackend();
        AutoencoderKL model = new AutoencoderKL(backend,
                new AutoencoderKL.Config(3, 3, 4, 8, 0.5f));
        Tensor image = TensorFactory.of(new float[3 * 8 * 8], backend, 1, 3, 8, 8);
        try {
            AutoencoderKL.LatentDistribution posterior = model.encode(image);
            Tensor first = posterior.sample(7L);
            Tensor second = posterior.sample(7L);
            try {
                assertArrayEquals(first.toFloatArray(), second.toFloatArray());
            } finally {
                first.release();
                second.release();
                posterior.mean().release();
                posterior.logVariance().release();
            }
        } finally {
            image.release();
            model.close();
        }
    }

    @Test
    void rejectsNonDivisibleSpatialDimensionsAndWrongChannels() {
        CpuBackend backend = new CpuBackend();
        AutoencoderKL model = new AutoencoderKL(backend,
                new AutoencoderKL.Config(3, 3, 4, 8, 0.5f));
        Tensor odd = TensorFactory.of(new float[3 * 7 * 8], backend, 1, 3, 7, 8);
        Tensor wrongChannels = TensorFactory.of(new float[4 * 8 * 8], backend, 1, 4, 8, 8);
        try {
            assertThrows(IllegalArgumentException.class, () -> model.encode(odd));
            assertThrows(IllegalArgumentException.class, () -> model.encode(wrongChannels));
        } finally {
            odd.release();
            wrongChannels.release();
            model.close();
        }
    }

    @Test
    void validatesScalingFactorAndChannelDimensions() {
        assertThrows(IllegalArgumentException.class,
                () -> new AutoencoderKL.Config(3, 3, 4, 8, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new AutoencoderKL.Config(3, 3, 0, 8, 0.5f));
    }
}
