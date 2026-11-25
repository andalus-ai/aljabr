/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantConfig.java
 * ───────────────────────
 * Quantization configuration.
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;

/**
 * Configuration for model quantization.
 *
 * @author bhangun
 * @version 1.0.0
 */
public class QuantConfig {

    /**
     * Target quantization strategy.
     */
    private final QuantizationStrategy strategy;

    /**
     * Group size for GPTQ quantization (e.g., 128).
     */
    private final int groupSize;

    /**
     * Bits for symmetric quantization.
     */
    private final int bits;

    /**
     * Whether to use symmetric quantization.
     */
    private final boolean symmetric;

    /**
     * Whether to use per-channel quantization.
     */
    private final boolean perChannel;

    /**
     * Act order for GPTQ (true for ActOrder, false for static).
     */
    private final boolean actOrder;

    /**
     * Dampening percentage for GPTQ (0.0 to 1.0).
     */
    private final double dampPercent;

    /**
     * Number of samples for calibration.
     */
    private final int numSamples;

    /**
     * Sequence length for calibration samples.
     */
    private final int seqLen;

    /**
     * Whether to use desc_act (descending activation order).
     */
    private final boolean descAct;

    /**
     * Default constructor with sensible defaults for INT4 GPTQ.
     */
    public QuantConfig() {
        this(QuantizationStrategy.INT4, 128, 4, false, true, false, 0.01, 128, 2048, false);
    }

    /**
     * Create quantization configuration.
     *
     * @param strategy    quantization strategy
     * @param groupSize   group size for quantization
     * @param bits        number of bits
     * @param symmetric   symmetric quantization
     * @param perChannel  per-channel quantization
     * @param actOrder    activation order for GPTQ
     * @param dampPercent dampening percentage
     * @param numSamples  number of calibration samples
     * @param seqLen      sequence length
     * @param descAct     descending activation order
     */
    public QuantConfig(QuantizationEngine.QuantStrategy strategy, int groupSize, int bits, boolean symmetric,
            boolean perChannel, boolean actOrder, double dampPercent, int numSamples, int seqLen, boolean descAct) {
        this(QuantizationStrategyCompatibility.fromLegacy(strategy), groupSize, bits, symmetric, perChannel,
                actOrder, dampPercent, numSamples, seqLen, descAct);
    }

    private QuantConfig(QuantizationStrategy strategy, int groupSize, int bits, boolean symmetric,
            boolean perChannel, boolean actOrder, double dampPercent, int numSamples, int seqLen,
            boolean descAct) {
        this.strategy = Objects.requireNonNull(strategy, "strategy cannot be null");
        this.groupSize = groupSize;
        this.bits = bits;
        this.symmetric = symmetric;
        this.perChannel = perChannel;
        this.actOrder = actOrder;
        this.dampPercent = dampPercent;
        this.numSamples = numSamples;
        this.seqLen = seqLen;
        this.descAct = descAct;
    }

    /**
     * Create builder for QuantConfig.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create default INT4 GPTQ config.
     *
     * @return INT4 GPTQ configuration
     */
    public static QuantConfig int4Gptq() {
        return builder().quantizationStrategy(QuantizationStrategy.INT4).build();
    }

    /**
     * Create default INT8 config.
     *
     * @return INT8 configuration
     */
    public static QuantConfig int8() {
        return builder().quantizationStrategy(QuantizationStrategy.INT8).bits(8).perChannel(true).build();
    }

    /**
     * Create default FP8 config.
     *
     * @return FP8 configuration
     */
    public static QuantConfig fp8() {
        return builder().quantizationStrategy(QuantizationStrategy.FP8).symmetric(true).build();
    }

    /**
     * Create AWQ configuration.
     *
     * @param bits      number of bits
     * @param groupSize group size
     * @return AWQ configuration
     */
    public static QuantConfig awq(int bits, int groupSize) {
        return builder().quantizationStrategy(QuantizationStrategy.AWQ).bits(bits).groupSize(groupSize).build();
    }

    /**
     * Create default TurboQuant configuration.
     *
     * @return TurboQuant configuration
     */
    public static QuantConfig turbo() {
        return builder().quantizationStrategy(QuantizationStrategy.TURBO).build();
    }

    /**
     * Create default BitsAndBytes configuration.
     *
     * @return BNB configuration
     */
    public static QuantConfig bnb() {
        return builder().quantizationStrategy(QuantizationStrategy.BNB).build();
    }

    /**
     * Create default AutoRound configuration.
     *
     * @return AutoRound configuration
     */
    public static QuantConfig autoRound() {
        return builder().quantizationStrategy(QuantizationStrategy.INT4).build();
    }

    public QuantizationEngine.QuantStrategy getStrategy() {
        return QuantizationStrategyCompatibility.toLegacy(strategy);
    }

    /**
     * Returns the neutral quantization strategy.
     */
    public QuantizationStrategy getQuantizationStrategy() {
        return strategy;
    }

    public int getGroupSize() {
        return groupSize;
    }

    public int getBits() {
        return bits;
    }

    public boolean isSymmetric() {
        return symmetric;
    }

    public boolean isPerChannel() {
        return perChannel;
    }

    public boolean isActOrder() {
        return actOrder;
    }

    public double getDampPercent() {
        return dampPercent;
    }

    public int getNumSamples() {
        return numSamples;
    }

    public int getSeqLen() {
        return seqLen;
    }

    public boolean isDescAct() {
        return descAct;
    }

    /**
     * Builder for QuantConfig.
     */
    public static class Builder {
        private QuantizationStrategy strategy = QuantizationStrategy.INT4;
        private int groupSize = 128;
        private int bits = 4;
        private boolean symmetric = false;
        private boolean perChannel = true;
        private boolean actOrder = false;
        private double dampPercent = 0.01;
        private int numSamples = 128;
        private int seqLen = 2048;
        private boolean descAct = false;

        /**
         * Compatibility entry point for callers using the engine-scoped strategy.
         */
        public Builder strategy(QuantizationEngine.QuantStrategy strategy) {
            return quantizationStrategy(QuantizationStrategy.fromLegacy(strategy));
        }

        /**
         * Sets the neutral quantization strategy.
         */
        public Builder quantizationStrategy(QuantizationStrategy strategy) {
            this.strategy = Objects.requireNonNull(strategy, "strategy cannot be null");
            if (strategy == QuantizationStrategy.INT8 || strategy == QuantizationStrategy.FP8) {
                this.bits = 8;
            } else if (strategy == QuantizationStrategy.INT4) {
                this.bits = 4;
            }
            return this;
        }

        public Builder groupSize(int groupSize) {
            this.groupSize = groupSize;
            return this;
        }

        public Builder bits(int bits) {
            this.bits = bits;
            return this;
        }

        public Builder symmetric(boolean symmetric) {
            this.symmetric = symmetric;
            return this;
        }

        public Builder perChannel(boolean perChannel) {
            this.perChannel = perChannel;
            return this;
        }

        public Builder actOrder(boolean actOrder) {
            this.actOrder = actOrder;
            return this;
        }

        public Builder dampPercent(double dampPercent) {
            this.dampPercent = dampPercent;
            return this;
        }

        public Builder numSamples(int numSamples) {
            this.numSamples = numSamples;
            return this;
        }

        public Builder seqLen(int seqLen) {
            this.seqLen = seqLen;
            return this;
        }

        public Builder descAct(boolean descAct) {
            this.descAct = descAct;
            return this;
        }

        public QuantConfig build() {
            return new QuantConfig(strategy, groupSize, bits, symmetric, perChannel, actOrder, dampPercent, numSamples,
                    seqLen, descAct);
        }
    }
}
