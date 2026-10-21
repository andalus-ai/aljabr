package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class TransformerEncoder extends NNModule {
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        public TransformerEncoder build() { return new TransformerEncoder(); }
        public Builder nLayers(int n) { return this; }
        public Builder dModel(int d) { return this; }
        public Builder numHeads(int h) { return this; }
        public Builder dFF(int d) { return this; }
        public Builder dropout(float d) { return this; }
        public Builder vocabSize(int d) { return this; }
        public Builder nHeads(int d) { return this; }
        public Builder maxSeqLen(int d) { return this; }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        return input;
    }
}
