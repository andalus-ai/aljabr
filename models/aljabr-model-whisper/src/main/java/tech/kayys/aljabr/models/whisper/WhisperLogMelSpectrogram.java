package tech.kayys.aljabr.models.whisper;

import java.util.Arrays;

/**
 * Pure-Java Whisper log-Mel feature extraction for one 30-second audio window.
 *
 * <p>Matches the Whisper feature-extractor geometry: 16 kHz mono input,
 * 400-sample periodic Hann window, 160-sample hop, 128 Slaney-normalized
 * Mel filters for large-v3, and 3000 output frames. The implementation uses
 * a Bluestein FFT so the exact 400-point transform is retained without
 * depending on a native DSP library.</p>
 *
 * <p>The returned matrix is laid out as {@code [melBin][frame]}. Input is
 * zero-padded or truncated to one 30-second window. Long recordings should be
 * split by {@link WhisperAudioPreprocessor}.</p>
 */
public final class WhisperLogMelSpectrogram {
    public static final int SAMPLE_RATE = WhisperLargeV3Config.EXPECTED_SAMPLE_RATE;
    public static final int FFT_SIZE = WhisperLargeV3Config.EXPECTED_FFT_SIZE;
    public static final int HOP_LENGTH = WhisperLargeV3Config.EXPECTED_HOP_LENGTH;
    public static final int MEL_BINS = WhisperLargeV3Config.EXPECTED_MEL_BINS;
    public static final int CHUNK_SAMPLES = SAMPLE_RATE * WhisperLargeV3Config.EXPECTED_CHUNK_LENGTH_SECONDS;
    public static final int FRAME_COUNT = CHUNK_SAMPLES / HOP_LENGTH;

    private static final int SPECTRUM_BINS = FFT_SIZE / 2 + 1;
    private static final double LOG_FLOOR = 1.0e-10;
    private static final double[] WINDOW = createWindow();
    private static final double[][] MEL_FILTERS = createMelFilters();
    private static final BluesteinFft FFT = new BluesteinFft(FFT_SIZE);

    public float[][] extract(float[] mono16KhzSamples) {
        if (mono16KhzSamples == null) {
            throw new IllegalArgumentException("audio samples must not be null");
        }
        if (mono16KhzSamples.length == 0) {
            throw new IllegalArgumentException("audio samples must not be empty");
        }

        float[] windowedAudio = new float[CHUNK_SAMPLES];
        System.arraycopy(mono16KhzSamples, 0, windowedAudio, 0,
                Math.min(mono16KhzSamples.length, CHUNK_SAMPLES));

        double[][] melPower = new double[MEL_BINS][FRAME_COUNT];
        double[] frameReal = new double[FFT_SIZE];
        double[] frameImaginary = new double[FFT_SIZE];

        for (int frame = 0; frame < FRAME_COUNT; frame++) {
            int frameStart = frame * HOP_LENGTH - FFT_SIZE / 2;
            for (int sample = 0; sample < FFT_SIZE; sample++) {
                int sourceIndex = reflectIndex(frameStart + sample, windowedAudio.length);
                frameReal[sample] = windowedAudio[sourceIndex] * WINDOW[sample];
                frameImaginary[sample] = 0.0;
            }

            FFT.transform(frameReal, frameImaginary);
            double[] power = new double[SPECTRUM_BINS];
            for (int bin = 0; bin < SPECTRUM_BINS; bin++) {
                power[bin] = frameReal[bin] * frameReal[bin]
                        + frameImaginary[bin] * frameImaginary[bin];
            }
            for (int mel = 0; mel < MEL_BINS; mel++) {
                double sum = 0.0;
                for (int bin = 0; bin < SPECTRUM_BINS; bin++) {
                    sum += MEL_FILTERS[mel][bin] * power[bin];
                }
                melPower[mel][frame] = Math.log10(Math.max(sum, LOG_FLOOR));
            }
        }

        double maximum = -Double.MAX_VALUE;
        for (double[] mel : melPower) {
            for (double value : mel) {
                maximum = Math.max(maximum, value);
            }
        }
        double clippingMaximum = maximum - 8.0;
        float[][] result = new float[MEL_BINS][FRAME_COUNT];
        for (int mel = 0; mel < MEL_BINS; mel++) {
            for (int frame = 0; frame < FRAME_COUNT; frame++) {
                double clipped = Math.min(melPower[mel][frame], clippingMaximum);
                result[mel][frame] = (float) ((clipped + 4.0) / 4.0);
            }
        }
        return result;
    }

    private static int reflectIndex(int index, int length) {
        if (length <= 1) {
            return 0;
        }
        int reflected = index;
        while (reflected < 0 || reflected >= length) {
            if (reflected < 0) {
                reflected = -reflected;
            }
            if (reflected >= length) {
                reflected = 2 * length - 2 - reflected;
            }
        }
        return reflected;
    }

    private static double[] createWindow() {
        double[] window = new double[FFT_SIZE];
        for (int i = 0; i < FFT_SIZE; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / FFT_SIZE);
        }
        return window;
    }

    private static double[][] createMelFilters() {
        double[][] filters = new double[MEL_BINS][SPECTRUM_BINS];
        double minMel = hzToSlaneyMel(0.0);
        double maxMel = hzToSlaneyMel(SAMPLE_RATE / 2.0);
        double[] melPoints = new double[MEL_BINS + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = slaneyMelToHz(minMel
                    + (maxMel - minMel) * i / (MEL_BINS + 1.0));
        }

        for (int mel = 0; mel < MEL_BINS; mel++) {
            double lower = melPoints[mel];
            double center = melPoints[mel + 1];
            double upper = melPoints[mel + 2];
            double enorm = 2.0 / (upper - lower);
            for (int bin = 0; bin < SPECTRUM_BINS; bin++) {
                double frequency = (double) bin * SAMPLE_RATE / FFT_SIZE;
                double lowerSlope = (frequency - lower) / (center - lower);
                double upperSlope = (upper - frequency) / (upper - center);
                filters[mel][bin] = Math.max(0.0, Math.min(lowerSlope, upperSlope)) * enorm;
            }
        }
        return filters;
    }

    // Librosa's default Slaney Mel scale, used by the Whisper reference frontend.
    private static double hzToSlaneyMel(double frequency) {
        double linearSpacing = 200.0 / 3.0;
        double minimumLogHz = 1000.0;
        double minimumLogMel = minimumLogHz / linearSpacing;
        double logStep = Math.log(6.4) / 27.0;
        if (frequency < minimumLogHz) {
            return frequency / linearSpacing;
        }
        return minimumLogMel + Math.log(frequency / minimumLogHz) / logStep;
    }

    private static double slaneyMelToHz(double mel) {
        double linearSpacing = 200.0 / 3.0;
        double minimumLogHz = 1000.0;
        double minimumLogMel = minimumLogHz / linearSpacing;
        double logStep = Math.log(6.4) / 27.0;
        if (mel < minimumLogMel) {
            return mel * linearSpacing;
        }
        return minimumLogHz * Math.exp(logStep * (mel - minimumLogMel));
    }

    /** Bluestein transform backed by an iterative radix-2 convolution FFT. */
    private static final class BluesteinFft {
        private final int size;
        private final int convolutionSize;
        private final double[] chirpCos;
        private final double[] chirpSin;
        private final double[] kernelReal;
        private final double[] kernelImaginary;

        private BluesteinFft(int size) {
            this.size = size;
            int required = 2 * size - 1;
            int powerOfTwo = 1;
            while (powerOfTwo < required) {
                powerOfTwo <<= 1;
            }
            convolutionSize = powerOfTwo;
            chirpCos = new double[size];
            chirpSin = new double[size];
            kernelReal = new double[convolutionSize];
            kernelImaginary = new double[convolutionSize];

            for (int i = 0; i < size; i++) {
                double phase = Math.PI * ((long) i * i % (2L * size)) / size;
                chirpCos[i] = Math.cos(phase);
                chirpSin[i] = Math.sin(phase);
                kernelReal[i] = chirpCos[i];
                kernelImaginary[i] = chirpSin[i];
                if (i != 0) {
                    kernelReal[convolutionSize - i] = chirpCos[i];
                    kernelImaginary[convolutionSize - i] = chirpSin[i];
                }
            }
            radix2(kernelReal, kernelImaginary, false);
        }

        private void transform(double[] real, double[] imaginary) {
            double[] workReal = new double[convolutionSize];
            double[] workImaginary = new double[convolutionSize];
            for (int i = 0; i < size; i++) {
                workReal[i] = real[i] * chirpCos[i] + imaginary[i] * chirpSin[i];
                workImaginary[i] = imaginary[i] * chirpCos[i] - real[i] * chirpSin[i];
            }

            radix2(workReal, workImaginary, false);
            for (int i = 0; i < convolutionSize; i++) {
                double realPart = workReal[i] * kernelReal[i]
                        - workImaginary[i] * kernelImaginary[i];
                double imaginaryPart = workReal[i] * kernelImaginary[i]
                        + workImaginary[i] * kernelReal[i];
                workReal[i] = realPart;
                workImaginary[i] = imaginaryPart;
            }
            radix2(workReal, workImaginary, true);

            for (int i = 0; i < size; i++) {
                real[i] = workReal[i] * chirpCos[i] + workImaginary[i] * chirpSin[i];
                imaginary[i] = workImaginary[i] * chirpCos[i] - workReal[i] * chirpSin[i];
            }
        }

        private static void radix2(double[] real, double[] imaginary, boolean inverse) {
            int n = real.length;
            for (int i = 1, j = 0; i < n; i++) {
                int bit = n >>> 1;
                for (; (j & bit) != 0; bit >>>= 1) {
                    j ^= bit;
                }
                j ^= bit;
                if (i < j) {
                    double temp = real[i];
                    real[i] = real[j];
                    real[j] = temp;
                    temp = imaginary[i];
                    imaginary[i] = imaginary[j];
                    imaginary[j] = temp;
                }
            }

            for (int length = 2; length <= n; length <<= 1) {
                double angle = (inverse ? 2.0 : -2.0) * Math.PI / length;
                double stepReal = Math.cos(angle);
                double stepImaginary = Math.sin(angle);
                int half = length >>> 1;
                for (int start = 0; start < n; start += length) {
                    double twiddleReal = 1.0;
                    double twiddleImaginary = 0.0;
                    for (int offset = 0; offset < half; offset++) {
                        int even = start + offset;
                        int odd = even + half;
                        double oddReal = real[odd] * twiddleReal
                                - imaginary[odd] * twiddleImaginary;
                        double oddImaginary = real[odd] * twiddleImaginary
                                + imaginary[odd] * twiddleReal;
                        real[odd] = real[even] - oddReal;
                        imaginary[odd] = imaginary[even] - oddImaginary;
                        real[even] += oddReal;
                        imaginary[even] += oddImaginary;

                        double nextReal = twiddleReal * stepReal
                                - twiddleImaginary * stepImaginary;
                        twiddleImaginary = twiddleReal * stepImaginary
                                + twiddleImaginary * stepReal;
                        twiddleReal = nextReal;
                    }
                }
            }

            if (inverse) {
                for (int i = 0; i < n; i++) {
                    real[i] /= n;
                    imaginary[i] /= n;
                }
            }
        }
    }
}
