package tech.kayys.aljabr.core.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Sequential container that chains modules together.
 */
public class Sequential extends NNModule {

    private final List<NNModule> moduleList = new ArrayList<>();

    /** @deprecated Prefer {@link #Sequential(ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Sequential() {
        super();
    }

    public Sequential(ComputeBackend backend) {
        super(backend);
    }

    /** @deprecated Prefer {@link #Sequential(ComputeBackend, NNModule...)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Sequential(NNModule... modules) {
        this(null, modules);
    }

    public Sequential(ComputeBackend backend, NNModule... modules) {
        super(backend);
        Objects.requireNonNull(modules, "modules must not be null");
        for (int i = 0; i < modules.length; i++) {
            add(String.valueOf(i), modules[i]);
        }
    }

    public Sequential add(String name, NNModule module) {
        registerModule(name, module);
        moduleList.add(module);
        return this;
    }

    public Sequential add(NNModule module) {
        return add(String.valueOf(moduleList.size()), module);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        Tensor output = input;
        for (NNModule module : moduleList) {
            output = module.forward(output);
        }
        return output;
    }

    public NNModule get(int index) {
        return moduleList.get(index);
    }

    public int size() {
        return moduleList.size();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Sequential(\n");
        for (int i = 0; i < moduleList.size(); i++) {
            sb.append("  (").append(i).append("): ")
                    .append(moduleList.get(i))
                    .append("\n");
        }
        return sb.append(")").toString();
    }
}
