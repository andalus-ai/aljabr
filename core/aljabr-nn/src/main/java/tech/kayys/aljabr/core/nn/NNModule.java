package tech.kayys.aljabr.core.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.lang.foreign.MemorySegment;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;

/**
 * Abstract base class for all neural network modules in Aljabr.
 * Manages parameters, buffers, and submodules.
 * @author bhangun
 */
public abstract class NNModule implements AutoCloseable {

    protected final Map<String, Tensor> parameters = new LinkedHashMap<>();
    protected final Map<String, Tensor> buffers = new LinkedHashMap<>();
    protected final Map<String, NNModule> submodules = new LinkedHashMap<>();
    protected boolean training = true;

    protected void registerParameter(String name, Tensor param) {
        Objects.requireNonNull(name, "Parameter name must not be null");
        parameters.put(name, param);
    }

    public Map<String, Tensor> parameters() {
        return parameters;
    }

    protected void registerBuffer(String name, Tensor buffer) {
        Objects.requireNonNull(name, "Buffer name must not be null");
        buffers.put(name, buffer);
    }

    protected <T extends NNModule> T registerModule(String name, T module) {
        Objects.requireNonNull(name, "NNModule name must not be null");
        Objects.requireNonNull(module, "NNModule must not be null");
        submodules.put(name, module);
        return module;
    }

    protected <T extends NNModule> T register(String name, T module) {
        Objects.requireNonNull(name, "Module name must not be null");
        submodules.put(name, module);
        return module;
    }


    protected Tensor register(String name, Tensor param) {
        registerParameter(name, param);
        return param;
    }

    public abstract Tensor forward(Tensor input);

    public NNModule train(boolean mode) {
        this.training = mode;
        submodules.values().forEach(m -> m.train(mode));
        return this;
    }

    public NNModule train() {
        return train(true);
    }

    public NNModule eval() {
        return train(false);
    }

    public boolean isTraining() {
        return training;
    }

    public NNModule to(DeviceType device) {
        parameters.replaceAll((name, param) -> param.to(device));
        buffers.replaceAll((name, buf) -> buf.to(device));
        submodules.values().forEach(m -> m.to(device));
        return this;
    }

    public void zeroGrad() {
        parameters.values().forEach(p -> {
            if (p.grad() != null) {
                p.setGrad(p.grad().zerosLike());
            }
        });
        submodules.values().forEach(NNModule::zeroGrad);
    }

    public List<Tensor> getParameters() {
        List<Tensor> all = new ArrayList<>(parameters.values());
        submodules.values().forEach(m -> all.addAll(m.getParameters()));
        return Collections.unmodifiableList(all);
    }

    public Map<String, Tensor> namedParameters() {
        Map<String, Tensor> result = new LinkedHashMap<>(parameters);
        submodules.forEach((name, module) -> module.namedParameters()
                .forEach((pName, param) -> result.put(name + "." + pName, param)));
        return Collections.unmodifiableMap(result);
    }

    public Map<String, NNModule> children() {
        return Collections.unmodifiableMap(submodules);
    }

    public long parameterCount() {
        long count = 0;
        for (Tensor p : parameters.values()) {
            count += p.shape().numel();
        }
        for (NNModule m : submodules.values()) {
            count += m.parameterCount();
        }
        return count;
    }

    /**
     * Returns an immutable mapping of fully qualified parameter and buffer names to
     * their live tensors. The map is a snapshot; tensor values are not cloned.
     */
    public Map<String, Tensor> stateDict() {
        Map<String, Tensor> result = new LinkedHashMap<>(namedParameters());
        namedBuffers().forEach((name, tensor) -> {
            if (result.putIfAbsent(name, tensor) != null) {
                throw new IllegalStateException("Duplicate state name: " + name);
            }
        });
        return Collections.unmodifiableMap(result);
    }

    /** Returns fully qualified registered buffers, including child modules. */
    public Map<String, Tensor> namedBuffers() {
        Map<String, Tensor> result = new LinkedHashMap<>(buffers);
        submodules.forEach((name, module) -> module.namedBuffers()
                .forEach((bufferName, tensor) -> result.put(name + "." + bufferName, tensor)));
        return Collections.unmodifiableMap(result);
    }

    /**
     * Loads a complete state dictionary. Strict mode rejects missing and unexpected
     * names. All names, shapes, dtypes, and storage compatibility are checked before
     * any tensor is mutated.
     */
    public void loadStateDict(Map<String, Tensor> stateDict) {
        loadStateDict(stateDict, true);
    }

    /**
     * Loads matching tensors from a state dictionary.
     *
     * @param stateDict source tensors keyed by fully qualified state name
     * @param strict when true, source and destination state names must match exactly
     * @throws IllegalArgumentException if names, shapes, or dtypes are incompatible
     * @throws UnsupportedOperationException if a tensor implementation cannot be copied in place
     */
    public void loadStateDict(Map<String, Tensor> stateDict, boolean strict) {
        Objects.requireNonNull(stateDict, "stateDict must not be null");
        Map<String, Tensor> current = stateDict();
        Map<String, Tensor> supplied = new LinkedHashMap<>();
        stateDict.forEach((name, tensor) -> {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("State names must not be null or blank");
            }
            supplied.put(name, Objects.requireNonNull(tensor, "State tensor must not be null: " + name));
        });

        if (strict) {
            List<String> missing = current.keySet().stream()
                    .filter(name -> !supplied.containsKey(name)).toList();
            List<String> unexpected = supplied.keySet().stream()
                    .filter(name -> !current.containsKey(name)).toList();
            if (!missing.isEmpty() || !unexpected.isEmpty()) {
                throw new IllegalArgumentException(
                        "State dictionary keys do not match; missing=" + missing + ", unexpected=" + unexpected);
            }
        }

        // Validate every matching entry before mutating any destination tensor.
        for (Map.Entry<String, Tensor> entry : supplied.entrySet()) {
            Tensor dst = current.get(entry.getKey());
            if (dst == null) {
                continue; // Only permitted in non-strict mode.
            }
            Tensor src = entry.getValue();
            if (!java.util.Arrays.equals(src.shape().dims(), dst.shape().dims())) {
                throw new IllegalArgumentException("Shape mismatch for state " + entry.getKey()
                        + ": expected " + dst.shape() + " but got " + src.shape());
            }
            if (src.dtype() != dst.dtype()) {
                throw new IllegalArgumentException("DType mismatch for state " + entry.getKey()
                        + ": expected " + dst.dtype() + " but got " + src.dtype());
            }
            if (!(src instanceof DefaultTensor) || !(dst instanceof DefaultTensor)) {
                throw new UnsupportedOperationException(
                        "In-place loadStateDict currently supports DefaultTensor only: " + entry.getKey());
            }
            long requiredBytes = dst.dtype().memoryFootprintBytes(dst.numel());
            if (((DefaultTensor) src).buffer().sizeBytes() < requiredBytes
                    || ((DefaultTensor) dst).buffer().sizeBytes() < requiredBytes) {
                throw new IllegalArgumentException("Insufficient storage for state " + entry.getKey()
                        + "; required " + requiredBytes + " bytes");
            }
        }

        for (Map.Entry<String, Tensor> entry : supplied.entrySet()) {
            Tensor dst = current.get(entry.getKey());
            if (dst == null) {
                continue;
            }
            long requiredBytes = dst.dtype().memoryFootprintBytes(dst.numel());
            MemorySegment srcSeg = ((DefaultTensor) entry.getValue()).buffer().segment();
            MemorySegment dstSeg = ((DefaultTensor) dst).buffer().segment();
            MemorySegment.copy(srcSeg, 0, dstSeg, 0, requiredBytes);
        }
    }

    @Override
    public void close() {
        parameters.values().forEach(Tensor::release);
        buffers.values().forEach(Tensor::release);
        submodules.values().forEach(NNModule::close);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(training=" + training + ")";
    }
}
