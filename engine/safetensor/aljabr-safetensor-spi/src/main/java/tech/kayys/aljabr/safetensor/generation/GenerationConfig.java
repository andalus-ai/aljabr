/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.generation;

import java.util.Collections;
import java.util.List;

/**
 * Immutable sampling configuration for an autoregressive generation session.
 * @author bhangun
 */
public final class GenerationConfig {

    public enum SamplingStrategy {
        GREEDY, TOP_K, TOP_P, TOP_K_TOP_P, BEAM
    }

    public enum KvCacheQuantization {
        NONE, INT8, INT4, TURBO
    }

    private final SamplingStrategy strategy;
    private final float temperature;
    private final int topK;
    private final float topP;
    private final float minP;
    private final int beamWidth;
    private final int maxNewTokens;
    private final int minNewTokens;
    private final List<Integer> stopTokenIds;
    private final List<String> stopStrings;
    private final float repetitionPenalty;
    private final float frequencyPenalty;
    private final boolean useKvCache;
    private final int maxKvCacheTokens;
    private final KvCacheQuantization kvCacheQuant;
    private final long seed;

    private GenerationConfig(Builder b) {
        this.strategy = b.strategy;
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.minP = b.minP;
        this.beamWidth = b.beamWidth;
        this.maxNewTokens = b.maxNewTokens;
        this.minNewTokens = b.minNewTokens;
        this.stopTokenIds = List.copyOf(b.stopTokenIds);
        this.stopStrings = List.copyOf(b.stopStrings);
        this.repetitionPenalty = b.repetitionPenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.useKvCache = b.useKvCache;
        this.maxKvCacheTokens = b.maxKvCacheTokens;
        this.kvCacheQuant = b.kvCacheQuant;
        this.seed = b.seed;
    }

    /**
     * Returns a default greedy config with {@code maxNewTokens=512}.
     *
     * @return a greedy {@link GenerationConfig}
     */
    public static GenerationConfig defaults() {
        return builder().strategy(SamplingStrategy.GREEDY).maxNewTokens(512).build();
    }

    /**
     * Creates a new builder with all defaults applied.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /** @return the sampling strategy */
    public SamplingStrategy strategy() { return strategy; }
    public float temperature() { return temperature; }
    public int topK() { return topK; }
    public float topP() { return topP; }
    public float minP() { return minP; }
    public int beamWidth() { return beamWidth; }
    public int maxNewTokens() { return maxNewTokens; }
    public int minNewTokens() { return minNewTokens; }
    public List<Integer> stopTokenIds() { return stopTokenIds; }
    public List<String> stopStrings() { return stopStrings; }
    public float repetitionPenalty() { return repetitionPenalty; }
    public float frequencyPenalty() { return frequencyPenalty; }
    public boolean useKvCache() { return useKvCache; }
    public int maxKvCacheTokens() { return maxKvCacheTokens; }
    public KvCacheQuantization kvCacheQuant() { return kvCacheQuant; }
    public long seed() { return seed; }

    /**
     * Returns {@code true} if the sampling strategy is {@link SamplingStrategy#GREEDY}.
     *
     * @return {@code true} for greedy decoding
     */
    public boolean isGreedy() {
        return strategy == SamplingStrategy.GREEDY;
    }

    /**
     * Returns {@code true} when the configured sampling parameters collapse to
     * deterministic argmax decoding.
     *
     * <p>This intentionally treats {@code topK == 1} and near-zero temperature as
     * greedy, because both settings select the highest logit deterministically even
     * if callers did not explicitly set {@link SamplingStrategy#GREEDY}.
     *
     * @return {@code true} when decoding should use argmax token selection
     */
    public boolean requestsGreedyDecoding() {
        return strategy == SamplingStrategy.GREEDY || temperature < 1.0e-4f || topK == 1;
    }

    /**
     * Returns {@code true} when greedy decoding can select directly from raw logits
     * without first applying mutable logit penalties.
     *
     * @return {@code true} for penalty-free greedy decoding
     */
    public boolean isPenaltyFreeGreedy() {
        return requestsGreedyDecoding()
                && repetitionPenalty == 1.0f
                && frequencyPenalty == 0.0f;
    }

    /**
     * Builder for {@link GenerationConfig}.
     */
    public static final class Builder {
        private SamplingStrategy strategy = SamplingStrategy.GREEDY;
        private float temperature = 1.0f;
        private int topK = 50;
        private float topP = 1.0f;
        private float minP = 0.0f;
        private int beamWidth = 1;
        private int maxNewTokens = 512;
        private int minNewTokens = 1;
        private List<Integer> stopTokenIds = Collections.emptyList();
        private List<String> stopStrings = Collections.emptyList();
        private float repetitionPenalty = 1.0f;
        private float frequencyPenalty = 0.0f;
        private boolean useKvCache = true;
        private int maxKvCacheTokens = 2048;
        private KvCacheQuantization kvCacheQuant = KvCacheQuantization.NONE;
        private long seed = -1L;

        /** @param v sampling strategy */
        public Builder strategy(SamplingStrategy v) { this.strategy = v; return this; }
        public Builder temperature(float v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(float v) { this.topP = v; return this; }
        public Builder minP(float v) { this.minP = v; return this; }
        public Builder beamWidth(int v) { this.beamWidth = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder minNewTokens(int v) { this.minNewTokens = v; return this; }
        public Builder stopTokenIds(List<Integer> v) { this.stopTokenIds = v; return this; }
        public Builder stopStrings(List<String> v) { this.stopStrings = v; return this; }
        public Builder repetitionPenalty(float v) { this.repetitionPenalty = v; return this; }
        public Builder frequencyPenalty(float v) { this.frequencyPenalty = v; return this; }
        public Builder useKvCache(boolean v) { this.useKvCache = v; return this; }
        public Builder maxKvCacheTokens(int v) { this.maxKvCacheTokens = v; return this; }
        public Builder kvCacheQuant(KvCacheQuantization v) { this.kvCacheQuant = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }

        /**
         * Builds the {@link GenerationConfig}.
         *
         * @return a new immutable {@link GenerationConfig}
         */
        public GenerationConfig build() {
            return new GenerationConfig(this);
        }
    }
}
