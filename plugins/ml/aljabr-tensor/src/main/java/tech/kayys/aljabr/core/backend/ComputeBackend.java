    default Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        throw new UnsupportedOperationException("RoPE is not supported by backend " + getClass().getName());
    }

    /**
     * Configurable rotary embedding operation for model-neutral NN modules.
     * Backends may override to support partial rotary dimensions and scaled positions.
     */
    default Tensor applyRoPE(Tensor input, int posOffset, float freqBase,
            int rotaryDimension, boolean isNeox, float positionScale) {
        if (rotaryDimension != input.shape().dim(3) || positionScale != 1.0f) {
            throw new UnsupportedOperationException("Configurable RoPE is not supported by backend " + getClass().getName());
        }
        return applyRoPE(input, posOffset, freqBase, isNeox);
    }