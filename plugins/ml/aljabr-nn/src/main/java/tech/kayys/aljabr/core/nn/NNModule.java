package tech.kayys.aljabr.core.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Abstract base class for all neural network modules in Aljabr.
 * Manages parameters, buffers, and submodules.
 * @author bhangun
 */
public abstract class NNModule implements AutoCloseable {

    protected final Map<String, Tensor> parameters = new LinkedHashMap<>();
    private final Map<String, Parameter> parameterObjects = new LinkedHashMap<>();
    private final java.util.Set<Tensor> retiredStateTensors =
            Collections.newSetFromMap(new java.util.IdentityHashMap<>());
    protected final Map<String, Tensor> buffers = new LinkedHashMap<>();
    protected final Map<String, NNModule> submodules = new LinkedHashMap<>();
    protected boolean training = true;
    private final ComputeBackend computeBackend;

    protected NNModule() {
        this(null);
    }

    /** Creates a module whose parameters and buffers are owned by the supplied backend. */
    protected NNModule(ComputeBackend computeBackend) {
        this.computeBackend = computeBackend;
    }

    /** Returns the explicitly configured backend, or {@code null} for legacy modules. */
    public final ComputeBackend computeBackend() {
        return computeBackend;
    }

    protected final Tensor randomNormal(long... shape) {
        return computeBackend == null ? TensorFactory.randn(shape) : TensorFactory.randn(computeBackend, shape);
    }

    protected final Tensor zeros(long... shape) {
        return computeBackend == null ? TensorFactory.zeros(shape) : TensorFactory.zeros(computeBackend, shape);
    }

    protected final Tensor ones(long... shape) {
        return computeBackend == null ? TensorFactory.ones(shape) : TensorFactory.ones(computeBackend, shape);
    }

    /** Creates tensor data on this module's backend, preserving legacy behavior for unconfigured modules. */
    @SuppressWarnings("removal")
    protected final Tensor tensorOf(float[] values, long... shape) {
        return computeBackend == null
                ? TensorFactory.of(values, shape)
                : TensorFactory.of(values, computeBackend, shape);
    }

    protected final void validateInputBackend(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        if (computeBackend != null && input.backend() != computeBackend) {
            throw new IllegalArgumentException("Module backend mismatch: expected "
                    + computeBackend.getClass().getName() + " but input uses "
                    + input.backend().getClass().getName());
        }
    }

    protected void registerParameter(String name, Tensor param) {
        registerParameter(name, param, true);
    }

    /** Registers a parameter and explicitly controls whether it is trainable. */
    protected void registerParameter(String name, Tensor param, boolean requiresGrad) {
        Objects.requireNonNull(name, "Parameter name must not be null");
        Objects.requireNonNull(param, "Parameter must not be null");
        validateOwnedTensor(name, param);
        param.setRequiresGrad(requiresGrad);
        Parameter registered = parameterObjects.get(name);
        if (registered == null) {
            parameters.put(name, param);
            parameterObjects.put(name, new Parameter(name, param, replacement -> replaceRegisteredParameter(name, replacement)));
        } else {
            Tensor previous = parameters.put(name, param);
            if (previous != null && previous != param) {
                retiredStateTensors.add(previous);
            }
            registered.replaceTensor(param);
        }
    }

    public Map<String, Tensor> parameters() {
        return parameters;
    }

    protected void registerBuffer(String name, Tensor buffer) {
        Objects.requireNonNull(name, "Buffer name must not be null");
        Objects.requireNonNull(buffer, "Buffer must not be null");
        validateOwnedTensor(name, buffer);
        buffers.put(name, buffer);
    }

    private void validateOwnedTensor(String name, Tensor tensor) {
        if (computeBackend != null && tensor.backend() != computeBackend) {
            throw new IllegalArgumentException("Module state '" + name + "' uses a different compute backend");
        }
    }

    protected <T extends NNModule> T registerModule(String name, T module) {
        Objects.requireNonNull(name, "NNModule name must not be null");
        Objects.requireNonNull(module, "NNModule must not be null");
        validateSubmoduleBackend(name, module);
        submodules.put(name, module);
        return module;
    }

    private void validateSubmoduleBackend(String name, NNModule module) {
        if (computeBackend == null) {
            return;
        }
        if (module.computeBackend() != null && module.computeBackend() != computeBackend) {
            throw new IllegalArgumentException("Submodule '" + name + "' uses a different compute backend");
        }
        module.stateDict().forEach((stateName, tensor) -> {
            if (tensor.backend() != computeBackend) {
                throw new IllegalArgumentException("Submodule '" + name + "' state '" + stateName
                        + "' uses a different compute backend");
            }
        });
    }

    protected <T extends NNModule> T register(String name, T module) {
        return registerModule(name, module);
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
        for (Map.Entry<String, Tensor> entry : new ArrayList<>(parameters.entrySet())) {
            String name = entry.getKey();
            Tensor previous = entry.getValue();
            Tensor moved = previous.to(device);
            Parameter registered = parameterObjects.get(name);
            if (registered != null) {
                registered.replaceTensor(moved);
            }
            parameters.put(name, moved);
            if (previous != moved) {
                retiredStateTensors.add(previous);
            }
        }
        for (Map.Entry<String, Tensor> entry : new ArrayList<>(buffers.entrySet())) {
            Tensor previous = entry.getValue();
            Tensor moved = previous.to(device);
            buffers.put(entry.getKey(), moved);
            if (previous != moved) {
                retiredStateTensors.add(previous);
            }
        }
        submodules.values().forEach(m -> m.to(device));
        return this;
    }

    /** Updates the module's live parameter map while retaining the previous value until close. */
    private void replaceRegisteredParameter(String name, Tensor replacement) {
        Tensor previous = parameters.put(name, replacement);
        if (previous != null && previous != replacement) {
            retiredStateTensors.add(previous);
        }
    }

    /**
     * Clears gradients for this module and all registered submodules.
     *
     * <p>Gradient tensors are owned by their parameter tensors. Replacing a
     * gradient therefore releases the previous gradient after the zeroed
     * replacement has been installed.</p>
     */
    public void zeroGrad() {
        parameters.values().forEach(parameter -> {
            Tensor previousGradient = parameter.grad();
            if (previousGradient == null) {
                return;
            }
            Tensor zeroGradient = previousGradient.zerosLike();
            parameter.setGrad(zeroGradient);
            if (previousGradient != zeroGradient) {
                previousGradient.release();
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

    /**
     * Returns registered parameter handles, including child modules, using
     * fully qualified names. The returned map is an immutable snapshot.
     *
     * <p>{@link #namedParameters()} remains available as the tensor-oriented
     * compatibility API.</p>
     */
    public Map<String, Parameter> namedParameterObjects() {
        Map<String, Parameter> result = new LinkedHashMap<>(parameterObjects);
        submodules.forEach((name, module) -> module.namedParameterObjects()
                .forEach((parameterName, parameter) ->
                        result.put(name + "." + parameterName, parameter)));
        return Collections.unmodifiableMap(result);
    }

    /** Returns unique parameter handles in module traversal order. */
    public List<Parameter> parameterObjects() {
        java.util.Set<Tensor> seen =
                Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        List<Parameter> result = new ArrayList<>();
        for (Parameter parameter : parameterObjects.values()) {
            if (seen.add(parameter.tensor())) {
                result.add(parameter);
            }
        }
        for (NNModule module : submodules.values()) {
            for (Parameter parameter : module.parameterObjects()) {
                if (seen.add(parameter.tensor())) {
                    result.add(parameter);
                }
            }
        }
        return Collections.unmodifiableList(result);
    }

    /** Returns parameters whose underlying tensors currently require gradients. */
    public List<Parameter> trainableParameterObjects() {
        return parameterObjects().stream().filter(Parameter::isTrainable).toList();
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
     * @throws UnsupportedOperationException if the destination backend does not support the required storage transfer
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
            dst.backend().validateCopyInto(dst, src);
        }

        for (Map.Entry<String, Tensor> entry : supplied.entrySet()) {
            Tensor dst = current.get(entry.getKey());
            if (dst == null) {
                continue;
            }
            dst.backend().copyInto(dst, entry.getValue());
        }
    }

    @Override
    public void close() {
        java.util.Set<Tensor> owned = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        owned.addAll(parameters.values());
        owned.addAll(buffers.values());
        owned.addAll(retiredStateTensors);
        owned.forEach(Tensor::release);
        retiredStateTensors.clear();
        parameters.clear();
        buffers.clear();
        submodules.values().forEach(NNModule::close);
        submodules.clear();
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(training=" + training + ")";
    }
}
