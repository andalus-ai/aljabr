package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;

/**
 * Float32 reference implementation of the published Gemma4 log-mel frontend.
 *
 * <p>Input is mono PCM already decoded and resampled to the configured sample
 * rate. Container decoding, channel mixing, and sample-rate conversion are
 * intentionally separate concerns. The implementation follows the upstream
 * semicausal framing, periodic Hann window, magnitude RFFT, unnormalized HTK
 * mel bank, and natural-log compression.</p>
 */
public final class EmbeddingGemma2AudioFeatureExtractor {
    private final int sampleRate;
    private final int featureSize;
    private final int fftLength;
    private final int frameLength;
    private final int hopLength;
    private final double minFrequency;
    private final double maxFrequency;
    private final double melFloor;
    private final double inputScaleFactor;
    private final double[] window;
    private final double[][] melFilters;

    public EmbeddingGemma2AudioFeatureExtractor(
            int sampleRate,
            int featureSize,
            int fftLength,
            int frameLength,
            int hopLength,
            double minFrequency,
            double maxFrequency,
            double melFloor,
            double inputScaleFactor) {
        if (sampleRate <= 0 || featureSize <= 0 || fftLength <= 0 || frameLength <= 0 || hopLength <= 0) {
            throw new IllegalArgumentException("audio frontend dimensions must be positive");
        }
        if (Integer.bitCount(fftLength) != 1 || frameLength > fftLength) {
            throw new IllegalArgumentException("fftLength must be a power of two >= frameLength");
        }
        if (!Double.isFinite(minFrequency) || !Double.isFinite(maxFrequency)
                || minFrequency < 0 || maxFrequency <= minFrequency || maxFrequency > sampleRate / 2.0) {
            throw new IllegalArgumentException("invalid mel frequency range");
        }
        if (!Double.isFinite(melFloor) || melFloor <= 0
                || !Double.isFinite(inputScaleFactor) || inputScaleFactor <= 0) {
            throw new IllegalArgumentException("melFloor and inputScaleFactor must be finite and positive");
        }
        this.sampleRate = sampleRate;
        this.featureSize = featureSize;
        this.fftLength = fftLength;
        this.frameLength = frameLength;
        this.hopLength = hopLength;
        this.minFrequency = minFrequency;
        this.maxFrequency = maxFrequency;
        this.melFloor = melFloor;
        this.inputScaleFactor = inputScaleFactor;
        this.window = periodicHann(frameLength);
        this.melFilters = createHtkMelFilterBank(fftLength / 2 + 1, featureSize,
                minFrequency, maxFrequency, sampleRate);
    }

    /** Published EmbeddingGemma 2 audio frontend settings (16 kHz mono PCM). */
    public static EmbeddingGemma2AudioFeatureExtractor publishedDefaults() {
        return new EmbeddingGemma2AudioFeatureExtractor(
                16_000, 128, 512, 320, 160, 0.0, 8_000.0, 0.001, 1.0);
    }

    /**
     * Extracts [melFrames][featureSize] features and one validity bit per frame.
     * The waveform is right-padded to a multiple of 128 samples, as in the
     * published feature-extractor call defaults.
     */
    public Features extract(float[] monoPcm, int inputSampleRate) {
        Objects.requireNonNull(monoPcm, "monoPcm must not be null");
        if (inputSampleRate != sampleRate) {
            throw new IllegalArgumentException("expected mono PCM at " + sampleRate
                    + " Hz; received " + inputSampleRate
                    + " Hz. Resample before feature extraction.");
        }
        if (monoPcm.length == 0) {
            throw new IllegalArgumentException("audio waveform must not be empty");
        }
        int paddedLength = Math.addExact(monoPcm.length, (128 - monoPcm.length % 128) % 128);
        int leftPadding = frameLength / 2;
        double[] padded = new double[Math.addExact(paddedLength, leftPadding)];
        boolean[] sampleMask = new boolean[padded.length];
        for (int i = 0; i < monoPcm.length; i++) {
            float sample = monoPcm[i];
            if (!Float.isFinite(sample)) {
                throw new IllegalArgumentException("waveform contains a non-finite sample at index " + i);
            }
            padded[leftPadding + i] = sample * inputScaleFactor;
            sampleMask[leftPadding + i] = true;
        }

        int frameSpan = frameLength + 1;
        int frameCount = padded.length < frameSpan ? 0
                : (padded.length - frameSpan) / hopLength + 1;
        if (frameCount == 0) {
            throw new IllegalArgumentException("waveform is too short to produce an audio frame");
        }

        float[][] features = new float[frameCount][featureSize];
        boolean[] frameMask = new boolean[frameCount];
        double[] real = new double[fftLength];
        double[] imaginary = new double[fftLength];
        double[] magnitude = new double[fftLength / 2 + 1];

        for (int frame = 0; frame < frameCount; frame++) {
            int start = frame * hopLength;
            java.util.Arrays.fill(real, 0.0);
            java.util.Arrays.fill(imaginary, 0.0);
            for (int i = 0; i < frameLength; i++) {
                real[i] = padded[start + i] * window[i];
            }
            fft(real, imaginary);
            for (int bin = 0; bin < magnitude.length; bin++) {
                magnitude[bin] = Math.hypot(real[bin], imaginary[bin]);
            }
            for (int mel = 0; mel < featureSize; mel++) {
                double energy = 0.0;
                for (int bin = 0; bin < magnitude.length; bin++) {
                    energy += magnitude[bin] * melFilters[bin][mel];
                }
                features[frame][mel] = (float) Math.log(energy + melFloor);
            }
            // Upstream marks a frame valid when its final analysis-window sample
            // belongs to the original, unpadded waveform.
            int finalSample = start + frameSpan - 1;
            frameMask[frame] = finalSample < sampleMask.length && sampleMask[finalSample];
            if (!frameMask[frame]) {
                java.util.Arrays.fill(features[frame], 0.0f);
            }
        }
        return new Features(features, frameMask, monoPcm.length, paddedLength);
    }

    public int sampleRate() { return sampleRate; }
    public int featureSize() { return featureSize; }

    private static double[] periodicHann(int length) {
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            result[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / length);
        }
        return result;
    }

    private static double[][] createHtkMelFilterBank(
            int bins, int filters, double minHz, double maxHz, int sampleRate) {
        double minMel = hertzToHtkMel(minHz);
        double maxMel = hertzToHtkMel(maxHz);
        double[] points = new double[filters + 2];
        for (int i = 0; i < points.length; i++) {
            double mel = minMel + (maxMel - minMel) * i / (filters + 1.0);
            points[i] = htkMelToHertz(mel);
        }
        double[][] bank = new double[bins][filters];
        for (int bin = 0; bin < bins; bin++) {
            double frequency = (sampleRate / 2.0) * bin / (bins - 1.0);
            for (int filter = 0; filter < filters; filter++) {
                double lower = (frequency - points[filter]) / (points[filter + 1] - points[filter]);
                double upper = (points[filter + 2] - frequency) / (points[filter + 2] - points[filter + 1]);
                bank[bin][filter] = Math.max(0.0, Math.min(lower, upper));
            }
        }
        return bank;
    }

    private static double hertzToHtkMel(double hz) {
        return 2595.0 * Math.log10(1.0 + hz / 700.0);
    }

    private static double htkMelToHertz(double mel) {
        return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
    }

    /** In-place radix-2 complex FFT. */
    private static void fft(double[] real, double[] imaginary) {
        int n = real.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >>> 1;
            for (; (j & bit) != 0; bit >>>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double r = real[i]; real[i] = real[j]; real[j] = r;
                double im = imaginary[i]; imaginary[i] = imaginary[j]; imaginary[j] = im;
            }
        }
        for (int size = 2; size <= n; size <<= 1) {
            int half = size >>> 1;
            double angle = -2.0 * Math.PI / size;
            double stepReal = Math.cos(angle);
            double stepImaginary = Math.sin(angle);
            for (int start = 0; start < n; start += size) {
                double wr = 1.0, wi = 0.0;
                for (int offset = 0; offset < half; offset++) {
                    int even = start + offset;
                    int odd = even + half;
                    double tr = wr * real[odd] - wi * imaginary[odd];
                    double ti = wr * imaginary[odd] + wi * real[odd];
                    real[odd] = real[even] - tr;
                    imaginary[odd] = imaginary[even] - ti;
                    real[even] += tr;
                    imaginary[even] += ti;
                    double nextWr = wr * stepReal - wi * stepImaginary;
                    wi = wr * stepImaginary + wi * stepReal;
                    wr = nextWr;
                }
            }
        }
    }

    /** Immutable result; arrays are defensively copied on construction and access. */
    public record Features(float[][] inputFeatures, boolean[] inputFeaturesMask,
                           int originalSampleCount, int paddedSampleCount) {
        public Features {
            Objects.requireNonNull(inputFeatures, "inputFeatures must not be null");
            Objects.requireNonNull(inputFeaturesMask, "inputFeaturesMask must not be null");
            if (inputFeatures.length != inputFeaturesMask.length) {
                throw new IllegalArgumentException("feature and mask frame counts differ");
            }
            float[][] copy = new float[inputFeatures.length][];
            for (int i = 0; i < copy.length; i++) copy[i] = inputFeatures[i].clone();
            inputFeatures = copy;
            inputFeaturesMask = inputFeaturesMask.clone();
        }
        @Override public float[][] inputFeatures() {
            float[][] copy = new float[inputFeatures.length][];
            for (int i = 0; i < copy.length; i++) copy[i] = inputFeatures[i].clone();
            return copy;
        }
        @Override public boolean[] inputFeaturesMask() { return inputFeaturesMask.clone(); }
    }
}
